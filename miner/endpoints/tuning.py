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
        
        # H100 x8 specifications
        self.gpu_memory_gb = 80  # H100 has 80GB VRAM
        self.gpu_count = 8
        self.total_memory_gb = 640  # 8x H100 = 640GB total
        
        # Task type weights (focus on high-value tasks)
        self.task_type_weights = {
            TaskType.INSTRUCTTEXTTASK: 0.35,  # 35% weight - Highest priority
            TaskType.GRPOTASK: 0.30,           # 30% weight - Second priority
            TaskType.IMAGETASK: 0.20,          # 20% weight - Third priority
            TaskType.DPOTASK: 0.15,            # 15% weight - Lower priority
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
        
        # Model family performance tracking (H100 x8 optimized)
        self.model_family_performance = {
            "llama": 0.95,    # Excellent on H100 x8
            "mistral": 0.90,  # Very good on H100 x8
            "qwen": 0.85,     # Good on H100 x8
            "gemma": 0.80,    # Decent on H100 x8
            "phi": 0.75,      # Acceptable on H100 x8
            "codellama": 0.90, # Excellent for code tasks
            "deepseek": 0.85,  # Good for reasoning
        }
        
        # Task type success rates (H100 x8 optimized)
        self.task_success_rates = {
            TaskType.INSTRUCTTEXTTASK: 0.98,  # Excellent on H100 x8
            TaskType.GRPOTASK: 0.95,          # Very good on H100 x8
            TaskType.IMAGETASK: 0.90,         # Good on H100 x8
            TaskType.DPOTASK: 0.85,           # Decent on H100 x8
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
        """Calculate job priority score for acceptance decision"""
        model_size = self.estimate_model_size(request.model)
        task_weight = self.task_type_weights.get(request.task_type, 0.1)
        
        # Model family performance
        model_family = self._get_model_family(request.model)
        model_performance = self.model_family_performance.get(model_family, 0.5)
        
        # Task success rate
        task_success = self.task_success_rates.get(request.task_type, 0.5)
        
        # Time efficiency (shorter jobs preferred)
        time_efficiency = 1.0 / max(request.hours_to_complete, 1)
        
        # Size efficiency (H100 x8 can handle larger models efficiently)
        size_efficiency = min(1.0, 70.0 / max(model_size, 1))  # Prefer larger models on H100 x8
        
        # H100 x8-specific bonuses
        h100_bonus = 0.15 if model_size > 13 else 0.10  # H100 x8 excels with large models
        
        # Priority formula optimized for H100 x8
        priority = (task_weight * 0.35 + 
                   model_performance * 0.25 + 
                   task_success * 0.20 + 
                   time_efficiency * 0.10 + 
                   size_efficiency * 0.05 + 
                   h100_bonus)
        
        return priority

    def _get_model_family(self, model_name: str) -> str:
        """Extract model family from model name"""
        model_name_lower = model_name.lower()
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
        return "unknown"

    def can_accept_job(self, request: MinerTaskOffer) -> bool:
        """Determine if we can accept this job based on current load and requirements"""
        # Check if we're already at capacity (H100 x8 can handle more concurrent jobs)
        if len(self.current_jobs) >= 12:  # Max 12 concurrent jobs with 8 H100s
            return False
        
        # Check if required GPUs are available
        model_size = self.estimate_model_size(request.model)
        required_gpus = self.get_optimal_gpu_allocation(model_size, request.task_type)
        
        # Simple availability check - in practice you'd track GPU usage
        return True

    def should_accept_job(self, request: MinerTaskOffer) -> tuple[bool, str]:
        """Advanced decision logic for job acceptance optimized for H100 x8"""
        
        # Priority 1: Task type filtering
        if request.task_type not in [TaskType.INSTRUCTTEXTTASK, TaskType.GRPOTASK, TaskType.IMAGETASK]:
            return False, f"Only accepting {TaskType.INSTRUCTTEXTTASK}, {TaskType.GRPOTASK}, and {TaskType.IMAGETASK} tasks"
        
        # Priority 2: Model family support (expanded for H100 x8)
        supported_families = ["llama", "mistral", "qwen", "gemma", "phi", "codellama", "deepseek"]
        model_family = self._get_model_family(request.model)
        if model_family not in supported_families:
            return False, f"Model family {model_family} not supported"
        
        # Priority 3: Capacity check
        if not self.can_accept_job(request):
            return False, "At capacity, cannot accept more jobs"
        
        # Priority 4: Time constraints (H100 x8 can handle longer jobs)
        if request.hours_to_complete > 20:  # H100 x8 can handle longer jobs
            return False, "Job duration too long (>20 hours)"
        
        # Priority 5: Calculate priority score
        priority_score = self.calculate_job_priority(request)
        
        # Accept if priority score is above threshold (lower threshold for H100 x8)
        if priority_score > 0.45:  # Lower threshold due to H100 x8 capabilities
            return True, f"Accepted with priority score {priority_score:.3f}"
        else:
            return False, f"Priority score {priority_score:.3f} below threshold"

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
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        repo_id = None
        
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
                repo_id = config_data.get("hub_model_id", None)
        else:
            config_filename = f"{task_id}.toml"
            config_path = os.path.join(cst.CONFIG_DIR, config_filename)
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
        summary="Get Training Repo",
        description="Retrieve the training repository and commit hash for the tournament.",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
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