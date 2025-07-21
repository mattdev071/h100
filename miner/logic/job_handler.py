import json
import os
import shutil
import uuid
from dataclasses import dataclass

import docker
import pandas as pd
import toml
import yaml
from docker.errors import DockerException
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi

from core import constants as cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import save_config_toml
from core.config.config_handler import update_flash_attention
from core.config.config_handler import update_model_info
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.docker_utils import stream_logs
from core.models.utility_models import DiffusionJob
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import TextDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import TextJob
from miner.utils import download_flux_unet

logger = get_logger(__name__)

# Advanced training configuration for maximum accuracy
class AccuracyOptimizedTrainingConfig:
    def __init__(self):
        # H100 x8 specifications
        self.gpu_memory_gb = 80
        self.gpu_count = 8
        self.total_memory_gb = 640
        
        # Model size to GPU allocation mapping for H100 x8
        self.gpu_allocation_map = {
            "small": (2, 2),     # 1-7B models: 2 H100s
            "medium": (2, 2),    # 7-13B models: 2 H100s
            "large": (2, 2),     # 13-70B models: 2 H100s
            "xlarge": (2, 2),    # 70B+ models: 2 H100s
        }
        
        # Advanced LoRA configurations optimized for accuracy
        self.lora_configs = {
            "small": {"r": 256, "alpha": 64, "dropout": 0.05},
            "medium": {"r": 512, "alpha": 128, "dropout": 0.1},
            "large": {"r": 1024, "alpha": 256, "dropout": 0.15},
            "xlarge": {"r": 2048, "alpha": 512, "dropout": 0.2},
        }
        
        # Learning rate schedules optimized for accuracy
        self.lr_schedules = {
            "small": {"lr": 4e-4, "warmup_ratio": 0.1, "scheduler": "cosine"},
            "medium": {"lr": 3e-4, "warmup_ratio": 0.15, "scheduler": "cosine"},
            "large": {"lr": 2e-4, "warmup_ratio": 0.2, "scheduler": "cosine"},
            "xlarge": {"lr": 1e-4, "warmup_ratio": 0.25, "scheduler": "cosine"},
        }
        
        # Batch size configurations optimized for accuracy
        self.batch_sizes = {
            "small": {"per_device": 16, "gradient_accumulation": 1},
            "medium": {"per_device": 8, "gradient_accumulation": 2},
            "large": {"per_device": 4, "gradient_accumulation": 4},
            "xlarge": {"per_device": 2, "gradient_accumulation": 8},
        }

    def get_model_size_category(self, model_size: int) -> str:
        """Determine model size category for H100 x8"""
        if model_size <= 7:
            return "small"
        elif model_size <= 13:
            return "medium"
        elif model_size <= 70:
            return "large"
        else:
            return "xlarge"

    def estimate_model_size(self, model_name: str) -> int:
        """Estimate model size from name"""
        import re
        model_name_lower = model_name.lower()
        
        # Extract size from model name
        size_match = re.search(r'(\d+)(b|B)', model_name_lower)
        if size_match:
            return int(size_match.group(1))
        
        # Default estimates
        if "llama" in model_name_lower:
            if "7b" in model_name_lower:
                return 7
            elif "13b" in model_name_lower:
                return 13
            elif "70b" in model_name_lower:
                return 70
            return 7
        elif "mistral" in model_name_lower:
            return 7
        elif "qwen" in model_name_lower:
            if "72b" in model_name_lower:
                return 72
            elif "14b" in model_name_lower:
                return 14
            return 7
        elif "gemma" in model_name_lower:
            return 2
        elif "phi" in model_name_lower:
            return 2
        elif "codellama" in model_name_lower:
            return 7
        elif "deepseek" in model_name_lower:
            return 7
        
        return 7

    def get_accuracy_optimized_config(self, model_name: str, task_type: str) -> dict:
        """Get accuracy-optimized configuration for H100 x8"""
        model_size = self.estimate_model_size(model_name)
        size_category = self.get_model_size_category(model_size)
        
        # Base configuration
        config = {
            "model_size": model_size,
            "size_category": size_category,
            "gpu_allocation": self.gpu_allocation_map[size_category],
            "lora_config": self.lora_configs[size_category],
            "lr_config": self.lr_schedules[size_category],
            "batch_config": self.batch_sizes[size_category],
        }
        
        # Task-specific accuracy optimizations
        if task_type == "instruct":
            # Optimize for instruction following and reasoning
            config["lora_config"]["r"] = min(config["lora_config"]["r"] * 1.4, 2048)
            config["lr_config"]["lr"] *= 1.2
            config["eval_steps"] = 50
            config["save_steps"] = 100
        elif task_type == "grpo":
            # Optimize for reward optimization (higher loss is better)
            config["lora_config"]["r"] = min(config["lora_config"]["r"] * 1.6, 2048)
            config["lr_config"]["lr"] *= 0.85  # Lower LR for stability
            config["eval_steps"] = 25
            config["save_steps"] = 50
        elif task_type == "dpo":
            # Optimize for preference learning
            config["lora_config"]["r"] = min(config["lora_config"]["r"] * 1.3, 2048)
            config["lr_config"]["lr"] *= 1.15
            config["eval_steps"] = 50
            config["save_steps"] = 100
        elif task_type == "image":
            # Optimize for visual quality
            config["lora_config"]["r"] = min(config["lora_config"]["r"] * 1.5, 2048)
            config["lr_config"]["lr"] *= 1.1
            config["eval_steps"] = 100
            config["save_steps"] = 200
        
        return config

# Global training config instance
training_config = AccuracyOptimizedTrainingConfig()


@dataclass
class DockerEnvironmentDiffusion:
    huggingface_token: str
    wandb_token: str
    job_id: str
    base_model: str

    def to_dict(self) -> dict[str, str]:
        return {
            "HUGGINGFACE_TOKEN": self.huggingface_token,
            "WANDB_TOKEN": self.wandb_token,
            "JOB_ID": self.job_id,
            "BASE_MODEL": self.base_model,
        }


@dataclass
class DockerEnvironment:
    huggingface_token: str
    wandb_token: str
    job_id: str
    dataset_type: str
    dataset_filename: str

    def to_dict(self) -> dict[str, str]:
        return {
            "HUGGINGFACE_TOKEN": self.huggingface_token,
            "WANDB_TOKEN": self.wandb_token,
            "JOB_ID": self.job_id,
            "DATASET_TYPE": self.dataset_type,
            "DATASET_FILENAME": self.dataset_filename,
        }


def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: TextDatasetType,
    file_format: FileFormat,
    task_id: str,
    expected_repo_name: str | None,
) -> dict:
    """
    Loads the config template and modifies it to create a new job config with accuracy optimizations.
    """
    if isinstance(dataset_type, InstructTextDatasetType | DpoDatasetType | ChatTemplateDatasetType):
        config_path = cst.CONFIG_TEMPLATE_PATH
    elif isinstance(dataset_type, GrpoDatasetType):
        config_path = cst.CONFIG_TEMPLATE_PATH_GRPO

    logger.info("Loading config template")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    # Get accuracy-optimized configuration
    task_type = "instruct" if isinstance(dataset_type, InstructTextDatasetType) else "grpo"
    optimized_config = training_config.get_accuracy_optimized_config(model, task_type)
    
    # Apply accuracy-optimized training configurations
    config = _apply_accuracy_optimizations(config, optimized_config, model, task_id, expected_repo_name)
    
    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions], task_id
            )
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]

    config = update_flash_attention(config, model)
    config = update_model_info(config, model, task_id, expected_repo_name)
    config["mlflow_experiment_name"] = dataset

    return config

def _apply_accuracy_optimizations(config: dict, optimized_config: dict, model: str, task_id: str, expected_repo_name: str) -> dict:
    """Apply accuracy-optimized training configurations"""
    
    # Advanced LoRA configuration for maximum accuracy
    lora_config = optimized_config["lora_config"]
    config["lora_r"] = lora_config["r"]
    config["lora_alpha"] = lora_config["alpha"]
    config["lora_dropout"] = lora_config["dropout"]
    
    # Advanced learning rate configuration for accuracy
    lr_config = optimized_config["lr_config"]
    config["learning_rate"] = lr_config["lr"]
    config["warmup_ratio"] = lr_config["warmup_ratio"]
    config["lr_scheduler"] = lr_config["scheduler"]
    
    # Advanced batch size configuration for accuracy
    batch_config = optimized_config["batch_config"]
    config["per_device_train_batch_size"] = batch_config["per_device"]
    config["gradient_accumulation_steps"] = batch_config["gradient_accumulation"]
    
    # H100 x8-specific training optimizations for accuracy
    config["bf16"] = True  # Use bfloat16 for H100 x8
    config["fp16"] = False
    config["gradient_checkpointing"] = True
    config["dataloader_pin_memory"] = True  # H100 x8 has enough memory
    
    # Advanced optimizer settings for accuracy
    config["optim"] = "adamw_torch"
    config["adam_beta1"] = 0.9
    config["adam_beta2"] = 0.999
    config["adam_epsilon"] = 1e-8
    config["weight_decay"] = 0.01
    
    # Advanced evaluation settings for accuracy
    config["evaluation_strategy"] = "steps"
    config["eval_steps"] = optimized_config.get("eval_steps", 25)
    config["save_steps"] = optimized_config.get("save_steps", 50)
    config["save_total_limit"] = 8  # Keep more checkpoints for accuracy
    
    # Advanced logging for accuracy monitoring
    config["logging_steps"] = 3  # More frequent logging
    config["report_to"] = "none"  # Disable wandb for faster training
    
    # Advanced data processing for accuracy
    config["dataloader_num_workers"] = 16  # More workers due to more memory
    config["remove_unused_columns"] = False
    
    # Advanced model settings for accuracy
    config["torch_compile"] = True  # Enable torch.compile for faster training
    config["ddp_find_unused_parameters"] = False
    
    # Advanced loss computation for accuracy
    config["label_smoothing_factor"] = 0.1
    
    # Advanced gradient clipping for accuracy
    config["max_grad_norm"] = 1.0
    
    # Advanced early stopping for accuracy
    config["load_best_model_at_end"] = True
    config["metric_for_best_model"] = "eval_loss"
    config["greater_is_better"] = False
    
    # H100 x8-specific optimizations for accuracy
    config["dataloader_prefetch_factor"] = 8  # Prefetch more data
    config["group_by_length"] = True  # Group by length for efficiency
    config["length_column_name"] = "length"
    config["max_length"] = 8192  # H100 x8 can handle longer sequences
    
    # Advanced LoRA settings for accuracy
    config["lora_target_modules"] = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config["lora_bias"] = "none"
    config["lora_task_type"] = "CAUSAL_LM"
    
    # H100 x8 memory optimizations for accuracy
    config["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    config["ddp_backend"] = "nccl"
    config["ddp_bucket_cap_mb"] = 50  # Larger buckets for H100 x8
    
    # Advanced training techniques for accuracy
    config["ema_decay"] = 0.9999  # Exponential Moving Average
    config["ema_update_every"] = 5  # Update every 5 steps
    
    # Advanced evaluation for accuracy
    config["eval_accumulation_steps"] = 4  # More accumulation for evaluation
    config["prediction_loss_only"] = False  # Get full evaluation metrics
    
    # Advanced regularization for accuracy
    config["attention_dropout"] = 0.1
    config["hidden_dropout"] = 0.1
    
    # Advanced data quality for accuracy
    config["max_seq_length"] = 8192
    config["pad_to_multiple_of"] = 8
    
    # Advanced monitoring for accuracy
    config["logging_first_step"] = True
    config["logging_dir"] = "./logs"
    config["run_name"] = f"accuracy_optimized_{task_id}"
    
    return config


def create_reward_funcs_file(reward_funcs: list[str], task_id: str, destination_dir: str = cst.CONFIG_DIR) -> list[str]:
    """
    Create a Python file with reward functions for GRPO training.

    Args:
        reward_funcs: List of strings containing Python reward function implementations
        task_id: Unique task identifier
    """
    filename = f"rewards_{task_id}"
    filepath = os.path.join(destination_dir, f"{filename}.py")

    func_names = []
    for reward_func in reward_funcs:
        if "def " in reward_func:
            func_name = reward_func.split("def ")[1].split("(")[0].strip()
            func_names.append(func_name)

    with open(filepath, "w") as f:
        f.write("# Auto-generated reward functions file\n\n")
        for reward_func in reward_funcs:
            f.write(f"{reward_func}\n\n")

    return filename, func_names


def _load_and_modify_config_diffusion(job: DiffusionJob) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    logger.info("Loading config template")
    if job.model_type == ImageModelType.SDXL:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = job.model
        config["train_data_dir"] = f"/dataset/images/{job.job_id}/img/"
        config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
        config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}"
    elif job.model_type == ImageModelType.FLUX:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = f"{cst.CONTAINER_FLUX_PATH}/flux_unet_{job.model.replace('/', '_')}.safetensors"
        config["train_data_dir"] = f"/dataset/images/{job.job_id}/img/"
        config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
        config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}"
    else:
        logger.error(f"Unknown model type: {job.model_type}")
    return config


def create_job_diffusion(
    job_id: str,
    model: str,
    dataset_zip: str,
    model_type: ImageModelType,
    expected_repo_name: str | None
):
    return DiffusionJob(
        job_id=job_id,
        model=model,
        dataset_zip=dataset_zip,
        model_type=model_type,
        expected_repo_name=expected_repo_name,
    )


def create_job_text(
    job_id: str,
    dataset: str,
    model: str,
    dataset_type: TextDatasetType,
    file_format: FileFormat,
    expected_repo_name: str | None,
):
    return TextJob(
        job_id=job_id,
        dataset=dataset,
        model=model,
        dataset_type=dataset_type,
        file_format=file_format,
        expected_repo_name=expected_repo_name,
    )


def start_tuning_container_diffusion(job: DiffusionJob):
    logger.info("=" * 80)
    logger.info("STARTING THE DIFFUSION TUNING CONTAINER")
    logger.info("=" * 80)

    config_path = os.path.join(cst.CONFIG_DIR, f"{job.job_id}.toml")

    config = _load_and_modify_config_diffusion(job)
    save_config_toml(config, config_path)

    logger.info(config)
    if job.model_type == ImageModelType.FLUX:
        logger.info(f"Downloading flux unet from {job.model}")
        flux_unet_path = download_flux_unet(job.model)

    prepare_dataset(
        training_images_zip_path=job.dataset_zip,
        training_images_repeat=(
            cst.DIFFUSION_SDXL_REPEATS if job.model_type == ImageModelType.SDXL
            else cst.DIFFUSION_FLUX_REPEATS
        ),
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=job.job_id,
    )

    docker_env = DockerEnvironmentDiffusion(
        huggingface_token=cst.HUGGINGFACE_TOKEN, wandb_token=cst.WANDB_TOKEN, job_id=job.job_id, base_model=job.model_type.value
    ).to_dict()
    logger.info(f"Docker environment: {docker_env}")

    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/dataset/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/dataset/outputs",
                "mode": "rw",
            },
            os.path.abspath(cst.DIFFUSION_DATASET_DIR): {
                "bind": "/dataset/images",
                "mode": "rw",
            },
        }

        if job.model_type == ImageModelType.FLUX:
            volume_bindings[os.path.dirname(flux_unet_path)] =  {
                "bind": cst.CONTAINER_FLUX_PATH,
                "mode": "rw",
            }

        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE_DIFFUSION,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
            detach=True,
            tty=True,
        )

        # Use the shared stream_logs function
        stream_logs(container)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        raise

    finally:
        if "container" in locals():
            container.remove(force=True)

        train_data_path = f"{cst.DIFFUSION_DATASET_DIR}/{job.job_id}"

        if os.path.exists(train_data_path):
            shutil.rmtree(train_data_path)


def _dpo_format_prompt(row, format_str):
    result = format_str
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_chosen(row, format_str):
    result = format_str
    if "{chosen}" in format_str and cst.DPO_DEFAULT_FIELD_CHOSEN in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_CHOSEN]):
        result = result.replace("{chosen}", str(row[cst.DPO_DEFAULT_FIELD_CHOSEN]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_rejected(row, format_str):
    result = format_str
    if "{rejected}" in format_str and cst.DPO_DEFAULT_FIELD_REJECTED in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_REJECTED]):
        result = result.replace("{rejected}", str(row[cst.DPO_DEFAULT_FIELD_REJECTED]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _adapt_columns_for_dpo_dataset(dataset_path: str, dataset_type: DpoDatasetType, apply_formatting: bool = False):
    """
    Transform a DPO JSON dataset file to match axolotl's `chatml.argilla` expected column names.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: DpoDatasetType with field mappings
        apply_formatting: If True, apply formatting templates to the content
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    column_mapping = {
        dataset_type.field_prompt: cst.DPO_DEFAULT_FIELD_PROMPT,
        dataset_type.field_system: cst.DPO_DEFAULT_FIELD_SYSTEM,
        dataset_type.field_chosen: cst.DPO_DEFAULT_FIELD_CHOSEN,
        dataset_type.field_rejected: cst.DPO_DEFAULT_FIELD_REJECTED
    }
    df = df.rename(columns=column_mapping)

    if apply_formatting:
        if dataset_type.prompt_format and dataset_type.prompt_format != "{prompt}":
            format_str = dataset_type.prompt_format
            df[cst.DPO_DEFAULT_FIELD_PROMPT] = df.apply(lambda row: _dpo_format_prompt(row, format_str), axis=1)
        if dataset_type.chosen_format and dataset_type.chosen_format != "{chosen}":
            format_str = dataset_type.chosen_format
            df[cst.DPO_DEFAULT_FIELD_CHOSEN] = df.apply(lambda row: _dpo_format_chosen(row, format_str), axis=1)
        if dataset_type.rejected_format and dataset_type.rejected_format != "{rejected}":
            format_str = dataset_type.rejected_format
            df[cst.DPO_DEFAULT_FIELD_REJECTED] = df.apply(lambda row: _dpo_format_rejected(row, format_str), axis=1)

    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def _adapt_columns_for_grpo_dataset(dataset_path: str, dataset_type: GrpoDatasetType):
    """
    Transform a GRPO JSON dataset file to match axolotl's `prompt` expected column name.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: GrpoDatasetType with field mappings
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.rename(columns={dataset_type.field_prompt: cst.GRPO_DEFAULT_FIELD_PROMPT})
    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def _create_docker_entrypoint(job):
    setup_commands = """
    echo 'Preparing data...' && \\
    if [ -n "$HUGGINGFACE_TOKEN" ]; then \\
    echo "Attempting to log in to Hugging Face" && \\
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential; \\
    else \\
    echo "HUGGINGFACE_TOKEN is not set. Skipping Hugging Face login."; \\
    fi && \\
    if [ -n "$WANDB_TOKEN" ]; then \\
    echo "Attempting to log in to W&B" && \\
    wandb login "$WANDB_TOKEN"; \\
    else \\
    echo "WANDB_TOKEN is not set. Skipping W&B login."; \\
    fi && \\
    if [ "$DATASET_TYPE" != "hf" ] && [ -f "/workspace/input_data/${DATASET_FILENAME}" ]; then \\
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/${DATASET_FILENAME}; \\
    fi"""

    if isinstance(job.dataset_type, GrpoDatasetType):
        reward_file = f"rewards_{job.job_id}.py"
        grpo_command = f"""
    echo "Moving specific reward function file to src directory..." && \\
    cp ${{CONFIG_DIR}}/{reward_file} /workspace/axolotl/src/"""
        setup_commands += " && \\" + grpo_command

    training_command = """
    echo 'Starting training command' && \\
    accelerate launch -m axolotl.cli.train ${CONFIG_DIR}/${JOB_ID}.yml
    """

    return setup_commands + " && \\" + training_command

def _adapt_columns_for_dataset(job: TextJob):
    """
    Adapt column names in the dataset based on job type.
    Only processes JSON files that require column name adaptation.
    """
    if job.file_format != FileFormat.JSON:
        return

    if isinstance(job.dataset_type, DpoDatasetType):
        _adapt_columns_for_dpo_dataset(job.dataset, job.dataset_type, True)
    elif isinstance(job.dataset_type, GrpoDatasetType):
        _adapt_columns_for_grpo_dataset(job.dataset, job.dataset_type)


def start_tuning_container(job: TextJob):
    logger.info("=" * 80)
    logger.info("STARTING THE TUNING CONTAINER")
    logger.info("=" * 80)

    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(cst.CONFIG_DIR, config_filename)

    docker_entrypoint = _create_docker_entrypoint(job)

    config = _load_and_modify_config(
        job.dataset,
        job.model,
        job.dataset_type,
        job.file_format,
        job.job_id,
        job.expected_repo_name,
    )
    save_config(config, config_path)

    logger.info(config)

    logger.info(os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "")

    docker_env = DockerEnvironment(
        huggingface_token=cst.HUGGINGFACE_TOKEN,
        wandb_token=cst.WANDB_TOKEN,
        job_id=job.job_id,
        dataset_type=cst.CUSTOM_DATASET_TYPE,
        dataset_filename=os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "",
    ).to_dict()
    logger.info(f"Docker environment: {docker_env}")

    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/workspace/axolotl/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/workspace/axolotl/outputs",
                "mode": "rw",
            },
        }

        if job.file_format != FileFormat.HF:
            dataset_dir = os.path.dirname(os.path.abspath(job.dataset))
            logger.info(dataset_dir)
            volume_bindings[dataset_dir] = {
                "bind": "/workspace/input_data",
                "mode": "ro",
            }

        _adapt_columns_for_dataset(job)

        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
            detach=True,
            tty=True,
            command=["/bin/bash", "-c", docker_entrypoint]
        )

        last_logs = stream_logs(container)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

    except Exception as e:
        # Waiting for axolotl to fix the issue
        if "TypeError: DPOTrainer.create_model_card() got an unexpected keyword argument 'dataset_tags'" in last_logs:
            logger.warning("This is probably just an axolotl issue only affecting HF repo model card, continuing...")
        else:
            logger.error(f"Error processing job: {str(e)}")
            raise

    finally:
        repo = config.get("hub_model_id", None)
        if repo:
            hf_api = HfApi(token=cst.HUGGINGFACE_TOKEN)
            hf_api.update_repo_visibility(repo_id=repo, private=False, token=cst.HUGGINGFACE_TOKEN)
            logger.info(f"Successfully made repository {repo} public")

        if "container" in locals():
            try:
                container.remove(force=True)
                logger.info("Container removed")
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")
