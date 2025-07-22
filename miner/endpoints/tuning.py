import os
import re
from datetime import datetime
from datetime import timedelta
from typing import Dict, List, Optional

import toml
import yaml
from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import get_config
from fiber.miner.dependencies import verify_get_request
from fiber.miner.dependencies import verify_request
from pydantic import ValidationError

import core.constants as cst
from core.models.payload_models import MinerTaskOffer
from core.models.payload_models import MinerTaskResponse
from core.models.utility_models import MinerSubmission
from validator.utils.hash_verification import calculate_model_hash

from core.models.payload_models import TrainingRepoResponse

from core.models.payload_models import TrainRequestGrpo
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.models.tournament_models import TournamentType
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job_diffusion
from miner.logic.job_handler import create_job_text

logger = get_logger(__name__)

# Advanced job selection strategy for 8x H100
class H100x8JobSelector:
    def __init__(self):
        self.current_jobs: Dict[str, datetime] = {}
        self.performance_history: Dict[str, float] = {}
        
        # Start automatic cleanup thread
        self._start_cleanup_thread()
        
        # H100 x8 specifications
        self.gpu_memory_gb = 80  # H100 has 80GB VRAM
        self.gpu_count = 8
        self.total_memory_gb = 640  # 8x H100 = 640GB total
        
        # Universal task type weights - accept all tasks equally
        self.task_type_weights = {
            TaskType.INSTRUCTTEXTTASK: 0.25,  # 25% weight - Equal priority
            TaskType.GRPOTASK: 0.25,           # 25% weight - Equal priority
            TaskType.IMAGETASK: 0.25,          # 25% weight - Equal priority
            TaskType.DPOTASK: 0.25,            # 25% weight - Equal priority
            TaskType.CHATTASK: 0.25,           # 25% weight - Equal priority
        }
        
        # GPU allocation strategy for 8x H100
        self.gpu_allocation = {
            "text_small": [0, 1],        # 1-7B models: 2 H100s
            "text_medium": [2, 3],       # 7-13B models: 2 H100s
            "text_large": [4, 5],        # 13-70B models: 2 H100s
            "text_xlarge": [6, 7],       # 70B+ models: 2 H100s
            "image_small": [0],           # Small image tasks: 1 H100
            "image_large": [1, 2],       # Large image tasks: 2 H100s
        }
        
        # Universal model family performance tracking (H100 x8 optimized)
        self.model_family_performance = {
            # Core families (excellent performance)
            "llama": 0.95,    # Excellent on H100 x8
            "mistral": 0.90,  # Very good on H100 x8
            "qwen": 0.85,     # Good on H100 x8
            "gemma": 0.80,    # Decent on H100 x8
            "phi": 0.75,      # Acceptable on H100 x8
            "codellama": 0.90, # Excellent for code tasks
            "deepseek": 0.85,  # Good for reasoning
            
            # Extended families (good performance)
            "gpt": 0.85,      # Good on H100 x8
            "bert": 0.80,     # Good for NLP tasks
            "t5": 0.80,       # Good for text generation
            "roberta": 0.80,  # Good for classification
            "falcon": 0.85,   # Good performance
            "mpt": 0.85,      # Good performance
            "opt": 0.80,      # Good performance
            "bloom": 0.80,    # Good performance
            "gpt2": 0.75,     # Acceptable performance
            "gptj": 0.80,     # Good performance
            "gptneo": 0.80,   # Good performance
            "gptneox": 0.80,  # Good performance
            "xglm": 0.80,     # Good performance
            "pythia": 0.80,   # Good performance
            "redpajama": 0.80, # Good performance
            "openllama": 0.85, # Good performance
            "vicuna": 0.85,   # Good performance
            "alpaca": 0.85,   # Good performance
            "wizard": 0.85,   # Good performance
            "baichuan": 0.85, # Good performance
            "chatglm": 0.85,  # Good performance
            "internlm": 0.85, # Good performance
            "yi": 0.85,       # Good performance
            "aquila": 0.85,   # Good performance
            "belle": 0.85,    # Good performance
            
            # Chinese model families
            "chinese-llama": 0.85,
            "chinese-alpaca": 0.85,
            "chinese-vicuna": 0.85,
            "chinese-baichuan": 0.85,
            "chinese-chatglm": 0.85,
            "chinese-internlm": 0.85,
            "chinese-yi": 0.85,
            "chinese-aquila": 0.85,
            "chinese-belle": 0.85,
            
            # Universal fallback
            "universal": 0.75, # Accept any model
        }
        
        # Universal task type success rates (H100 x8 optimized)
        self.task_success_rates = {
            TaskType.INSTRUCTTEXTTASK: 0.95,  # Excellent on H100 x8
            TaskType.GRPOTASK: 0.95,          # Excellent on H100 x8
            TaskType.IMAGETASK: 0.90,         # Good on H100 x8
            TaskType.DPOTASK: 0.90,           # Good on H100 x8
            TaskType.CHATTASK: 0.90,          # Good on H100 x8
        }

    def estimate_model_size(self, model_name: str) -> int:
        """Estimate model size in billions of parameters"""
        model_name_lower = model_name.lower()
        
        # Extract size from model name
        size_match = re.search(r'(\d+)(b|B)', model_name_lower)
        if size_match:
            return int(size_match.group(1))
        
        # Default estimates based on model family
        if "llama" in model_name_lower:
            if "7b" in model_name_lower or "7B" in model_name_lower:
                return 7
            elif "13b" in model_name_lower or "13B" in model_name_lower:
                return 13
            elif "70b" in model_name_lower or "70B" in model_name_lower:
                return 70
            else:
                return 7  # Default for llama
        elif "mistral" in model_name_lower:
            return 7
        elif "qwen" in model_name_lower:
            if "72b" in model_name_lower:
                return 72
            elif "14b" in model_name_lower:
                return 14
            else:
                return 7
        elif "gemma" in model_name_lower:
            return 2
        elif "phi" in model_name_lower:
            return 2
        elif "codellama" in model_name_lower:
            return 7
        elif "deepseek" in model_name_lower:
            return 7
        
        return 7  # Default fallback

    def get_optimal_gpu_allocation(self, model_size: int, task_type: TaskType) -> List[int]:
        """Determine optimal GPU allocation based on model size and task type"""
        if task_type == TaskType.IMAGETASK:
            if model_size <= 7:
                return self.gpu_allocation["image_small"]
            else:
                return self.gpu_allocation["image_large"]
        
        if model_size <= 7:
            return self.gpu_allocation["text_small"]
        elif model_size <= 13:
            return self.gpu_allocation["text_medium"]
        elif model_size <= 70:
            return self.gpu_allocation["text_large"]
        else:
            return self.gpu_allocation["text_xlarge"]

    def calculate_job_priority(self, request: MinerTaskOffer) -> float:
        """Accuracy-optimized job priority calculation"""
        model_size = self.estimate_model_size(request.model)
        task_weight = self.task_type_weights.get(request.task_type, 0.25)  # Default to equal weight
        
        # Accuracy-focused model family performance
        model_family = self._get_model_family(request.model)
        model_performance = self.model_family_performance.get(model_family, 0.75)  # Default to good performance
        
        # Accuracy-focused task success rate
        task_success = self.task_success_rates.get(request.task_type, 0.90)  # Default to good success rate
        
        # Time efficiency (shorter jobs preferred for accuracy)
        time_efficiency = 1.0 / max(request.hours_to_complete, 1)
        
        # Size efficiency (H100 x8 with 2 GPUs per job for accuracy)
        size_efficiency = min(1.0, 100.0 / max(model_size, 1))  # H100 x8 can handle any model
        
        # Accuracy-focused H100 x8 bonuses
        h100_accuracy_bonus = 0.25  # H100 x8 accuracy advantage
        accuracy_bonus = 0.30  # High bonus for accuracy focus
        
        # Accuracy-optimized priority formula
        priority = (task_weight * 0.20 + 
                   model_performance * 0.20 + 
                   task_success * 0.20 + 
                   time_efficiency * 0.10 + 
                   size_efficiency * 0.05 + 
                   h100_accuracy_bonus + 
                   accuracy_bonus)
        
        # Ensure high priority for accuracy optimization
        return min(max(priority, 0.70), 1.0)  # Minimum 70% priority for accuracy

    def _get_model_family(self, model_name: str) -> str:
        """Universal model family detection - support any model"""
        model_name_lower = model_name.lower()
        
        # Extended model family detection
        if "llama" in model_name_lower:
            return "llama"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "qwen" in model_name_lower:
            return "qwen"
        elif "gemma" in model_name_lower:
            return "gemma"
        elif "phi" in model_name_lower:
            return "phi"
        elif "codellama" in model_name_lower:
            return "codellama"
        elif "deepseek" in model_name_lower:
            return "deepseek"
        elif "gpt" in model_name_lower:
            return "gpt"
        elif "bert" in model_name_lower:
            return "bert"
        elif "t5" in model_name_lower:
            return "t5"
        elif "roberta" in model_name_lower:
            return "roberta"
        elif "falcon" in model_name_lower:
            return "falcon"
        elif "mpt" in model_name_lower:
            return "mpt"
        elif "opt" in model_name_lower:
            return "opt"
        elif "bloom" in model_name_lower:
            return "bloom"
        elif "gpt2" in model_name_lower:
            return "gpt2"
        elif "gptj" in model_name_lower:
            return "gptj"
        elif "gptneo" in model_name_lower:
            return "gptneo"
        elif "gptneox" in model_name_lower:
            return "gptneox"
        elif "xglm" in model_name_lower:
            return "xglm"
        elif "mpt" in model_name_lower:
            return "mpt"
        elif "pythia" in model_name_lower:
            return "pythia"
        elif "redpajama" in model_name_lower:
            return "redpajama"
        elif "openllama" in model_name_lower:
            return "openllama"
        elif "vicuna" in model_name_lower:
            return "vicuna"
        elif "alpaca" in model_name_lower:
            return "alpaca"
        elif "wizard" in model_name_lower:
            return "wizard"
        elif "baichuan" in model_name_lower:
            return "baichuan"
        elif "chatglm" in model_name_lower:
            return "chatglm"
        elif "internlm" in model_name_lower:
            return "internlm"
        elif "yi" in model_name_lower:
            return "yi"
        elif "aquila" in model_name_lower:
            return "aquila"
        elif "belle" in model_name_lower:
            return "belle"
        elif "chinese-llama" in model_name_lower:
            return "chinese-llama"
        elif "chinese-alpaca" in model_name_lower:
            return "chinese-alpaca"
        elif "chinese-vicuna" in model_name_lower:
            return "chinese-vicuna"
        elif "chinese-baichuan" in model_name_lower:
            return "chinese-baichuan"
        elif "chinese-chatglm" in model_name_lower:
            return "chinese-chatglm"
        elif "chinese-internlm" in model_name_lower:
            return "chinese-internlm"
        elif "chinese-yi" in model_name_lower:
            return "chinese-yi"
        elif "chinese-aquila" in model_name_lower:
            return "chinese-aquila"
        elif "chinese-belle" in model_name_lower:
            return "chinese-belle"
        else:
            # Universal fallback - accept any model
            return "universal"

    def _start_cleanup_thread(self):
        """Start automatic cleanup thread for expired jobs"""
        import threading
        import time
        
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_expired_jobs()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
                    time.sleep(60)  # Continue even if there's an error
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Started automatic job cleanup thread")

    def _cleanup_expired_jobs(self):
        """Remove expired jobs from current_jobs tracking"""
        from datetime import datetime
        
        current_time = datetime.now()
        expired_jobs = []
        
        for task_id, finish_time in self.current_jobs.items():
            if current_time > finish_time:
                expired_jobs.append(task_id)
        
        if expired_jobs:
            for task_id in expired_jobs:
                del self.current_jobs[task_id]
                logger.info(f"Automatically cleaned up expired job: {task_id}")
            
            logger.info(f"Cleaned up {len(expired_jobs)} expired jobs. Current jobs: {len(self.current_jobs)}/4")

    def cleanup_expired_jobs(self):
        """Manual cleanup method for immediate job removal"""
        self._cleanup_expired_jobs()

    def can_accept_job(self, request: MinerTaskOffer) -> bool:
        """Universal capacity management - H100 x8 can handle many jobs"""
        # Clean up expired jobs before checking capacity
        self._cleanup_expired_jobs()
        
        # Accuracy-optimized capacity for H100 x8
        max_concurrent_jobs = 4  # Optimal for accuracy (2 H100s per job)
        
        # Check if we're at accuracy-optimized capacity
        if len(self.current_jobs) >= max_concurrent_jobs:
            logger.warning(f"At accuracy capacity: {len(self.current_jobs)}/{max_concurrent_jobs} jobs")
            return False
        
        # H100 x8 with 2 GPUs per job for optimal accuracy
        model_size = self.estimate_model_size(request.model)
        logger.info(f"Accuracy capacity check: {len(self.current_jobs)}/{max_concurrent_jobs} jobs, model size: {model_size}B")
        
        # Accept jobs within accuracy-optimized capacity limits
        return True

    def should_accept_job(self, request: MinerTaskOffer) -> tuple[bool, str]:
        """Accuracy-optimized job acceptance - 4 jobs for maximum accuracy"""
        
        # Accept ALL task types - no filtering
        logger.info(f"Evaluating job for accuracy: {request.task_type} - {request.model} - {request.hours_to_complete}h")
        
        # Priority 1: Accuracy-optimized capacity check (4 jobs for accuracy)
        if not self.can_accept_job(request):
            return False, "At accuracy-optimized capacity (4 jobs)"
        
        # Priority 2: Time constraints (longer jobs for accuracy)
        if request.hours_to_complete > 72:  # Extended for accuracy optimization
            return False, "Job duration too long (>72 hours)"
        
        # Priority 3: Calculate accuracy-optimized priority
        priority_score = self.calculate_job_priority(request)
        
        # Accept if priority score is high (accuracy focus)
        if priority_score > 0.70:  # Higher threshold for accuracy
            logger.info(f"Accepting job for accuracy optimization - priority {priority_score:.3f}")
            return True, f"Accepted for accuracy optimization - priority {priority_score:.3f}"
        else:
            logger.info(f"Priority {priority_score:.3f} below accuracy threshold")
            return False, f"Priority {priority_score:.3f} below accuracy threshold"

# Global job selector instance
job_selector = H100x8JobSelector()

current_job_finish_time = None


async def tune_model_text(
    train_request: TrainRequestText,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")

    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    logger.info(f"Job received is {train_request}")

    try:
        logger.info(train_request.file_format)
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(train_request.dataset)
                train_request.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_text(
        job_id=str(train_request.task_id),
        dataset=train_request.dataset,
        model=train_request.model,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def tune_model_grpo(
    train_request: TrainRequestGrpo,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")

    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    logger.info(f"Job received is {train_request}")

    try:
        logger.info(train_request.file_format)
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(train_request.dataset)
                train_request.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_text(
        job_id=str(train_request.task_id),
        dataset=train_request.dataset,
        model=train_request.model,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def tune_model_diffusion(
    train_request: TrainRequestImage,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")

    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    logger.info(f"Job received is {train_request}")
    try:
        train_request.dataset_zip = await download_s3_file(
            train_request.dataset_zip, f"{cst.DIFFUSION_DATASET_DIR}/{train_request.task_id}.zip"
        )
        logger.info(train_request.dataset_zip)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_diffusion(
        job_id=str(train_request.task_id),
        dataset_zip=train_request.dataset_zip,
        model=train_request.model,
        model_type=train_request.model_type,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def get_latest_model_submission(task_id: str) -> MinerSubmission:
    try:
        config_filename = f"{task_id}.yml"
        # Use absolute path to config directory
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "core", "config")
        config_path = os.path.join(config_dir, config_filename)
        repo_id = None
        
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
                repo_id = config_data.get("hub_model_id", None)
        else:
            config_filename = f"{task_id}.toml"
            config_path = os.path.join(config_dir, config_filename)
            with open(config_path, "r") as file:
                config_data = toml.load(file)
                repo_id = config_data.get("huggingface_repo_id", None)

        if repo_id is None:
            raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")

        model_hash = calculate_model_hash(repo_id)
        
        return MinerSubmission(repo=repo_id, model_hash=model_hash)

    except FileNotFoundError as e:
        logger.error(f"No submission found for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")
    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving latest model submission: {str(e)}",
        )


async def task_offer(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info("H100 x8 advanced job offer evaluation")
        
        # Use H100 x8-optimized job selector
        should_accept, reason = job_selector.should_accept_job(request)
        
        if should_accept:
            # Track the job
            job_selector.current_jobs[request.task_id] = datetime.now() + timedelta(hours=request.hours_to_complete)
            logger.info(f"Accepting job: {reason}")
            return MinerTaskResponse(message=f"Accepted: {reason}", accepted=True)
        else:
            logger.info(f"Rejecting job: {reason}")
            return MinerTaskResponse(message=f"Rejected: {reason}", accepted=False)

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def task_offer_image(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info("H100 x8 advanced image job offer evaluation")
        
        # Use H100 x8-optimized job selector for image tasks
        should_accept, reason = job_selector.should_accept_job(request)
        
        if should_accept:
            # Track the job
            job_selector.current_jobs[request.task_id] = datetime.now() + timedelta(hours=request.hours_to_complete)
            logger.info(f"Accepting image job: {reason}")
            return MinerTaskResponse(message=f"Accepted: {reason}", accepted=True)
        else:
            logger.info(f"Rejecting image job: {reason}")
            return MinerTaskResponse(message=f"Rejected: {reason}", accepted=False)

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/rayonlabs/G.O.D", commit_hash="076e87fc746985e272015322cc91fb3bbbca2f26"
    )


async def get_job_status():
    """Get current job status and capacity information"""
    from datetime import datetime
    
    # Clean up expired jobs first
    job_selector.cleanup_expired_jobs()
    
    current_jobs = job_selector.current_jobs
    current_time = datetime.now()
    
    # Calculate job details
    job_details = []
    for task_id, finish_time in current_jobs.items():
        remaining_time = finish_time - current_time
        hours_remaining = remaining_time.total_seconds() / 3600
        
        job_details.append({
            "task_id": task_id,
            "finish_time": finish_time.isoformat(),
            "hours_remaining": max(0, hours_remaining),
            "status": "running" if hours_remaining > 0 else "expired"
        })
    
    return {
        "current_jobs": len(current_jobs),
        "max_capacity": 4,
        "available_slots": 4 - len(current_jobs),
        "can_accept_jobs": len(current_jobs) < 4,
        "jobs": job_details
    }


async def cleanup_jobs():
    """Manually trigger job cleanup"""
    job_selector.cleanup_expired_jobs()
    
    return {
        "message": "Job cleanup completed",
        "current_jobs": len(job_selector.current_jobs),
        "available_slots": 4 - len(job_selector.current_jobs)
    }


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer/",
        task_offer,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/task_offer_image/",
        task_offer_image,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=MinerSubmission,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    router.add_api_route(
        "/training_repo/{task_type}",
        get_training_repo,
        tags=["Subnet"],
        methods=["GET"],
        response_model=TrainingRepoResponse,
    )

    router.add_api_route(
        "/job_status",
        get_job_status,
        tags=["Status"],
        methods=["GET"],
        summary="Get Job Status",
        description="Get current job status and capacity information",
    )

    router.add_api_route(
        "/cleanup_jobs",
        cleanup_jobs,
        tags=["Status"],
        methods=["POST"],
        summary="Cleanup Jobs",
        description="Manually trigger cleanup of expired jobs",
    )

    router.add_api_route(
        "/start_training/",  # TODO: change to /start_training_text or similar
        tune_model_text,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/start_training_grpo/",
        tune_model_grpo,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/start_training_image/",
        tune_model_diffusion,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    return router