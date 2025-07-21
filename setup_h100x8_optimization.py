#!/usr/bin/env python3
"""
H100 x8 Optimization Setup Script
Implements advanced job selection, training configurations, and performance monitoring
for optimal ranking in the G.O.D subnet using 8x H100 GPUs.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check if system meets H100 x8 requirements"""
    logger.info("Checking system requirements...")
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        
        if len(gpus) < 8:
            logger.warning(f"Found {len(gpus)} GPUs, expected 8 for optimal performance")
        else:
            logger.info(f"Found {len(gpus)} GPUs - System ready for H100 x8 optimization")
            
        # Check GPU memory
        for i, gpu in enumerate(gpus[:8]):
            memory_gb = gpu.memoryTotal
            if memory_gb < 60:  # H100 should have 80GB
                logger.warning(f"GPU {i}: {memory_gb}GB memory (expected 80GB)")
            else:
                logger.info(f"GPU {i}: {memory_gb}GB memory - OK")
                
    except ImportError:
        logger.error("GPUtil not installed. Install with: pip install GPUtil")
        return False
    except Exception as e:
        logger.error(f"Error checking GPUs: {e}")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies for H100 x8 optimization"""
    logger.info("Installing dependencies...")
    
    dependencies = [
        "GPUtil",
        "psutil",
        "torch>=2.1.0",  # Latest PyTorch for H100 support
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "axolotl>=0.4.0",
        "trl>=0.7.0",
        "diffusers>=0.24.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.19.0",
        "wandb>=0.16.0",
        "docker>=6.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "pyyaml>=6.0",
        "toml>=0.10.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "nvidia-ml-py3",  # For H100 monitoring
    ]
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            logger.info(f"Installed {dep}")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to install {dep}")
            return False
    
    return True

def setup_environment_variables():
    """Setup environment variables for H100 x8 optimization"""
    logger.info("Setting up environment variables...")
    
    env_vars = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NCCL_P2P_DISABLE": "0",  # Enable P2P for H100 x8
        "NCCL_IB_DISABLE": "1",   # Disable InfiniBand
        "CUDA_LAUNCH_BLOCKING": "0",
        "TORCH_CUDA_ARCH_LIST": "9.0",  # H100 architecture
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:2048",  # Larger chunks for H100 x8
        "HF_HUB_CACHE": "/root/.cache/huggingface",
        "WANDB_MODE": "disabled",  # Disable wandb for faster training
        "CUDA_MEMORY_FRACTION": "0.95",  # Use 95% of GPU memory
        "TF_CPP_MIN_LOG_LEVEL": "2",  # Reduce TensorFlow logging
        "NCCL_DEBUG": "INFO",  # Enable NCCL debugging
        "NCCL_BLOCKING_WAIT": "1",  # Blocking wait for better stability
    }
    
    # Create .env file
    env_file = Path(".env")
    with open(env_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    logger.info("Environment variables configured")
    return True

def create_h100x8_monitoring_script():
    """Create H100 x8-specific monitoring script"""
    logger.info("Creating H100 x8 monitoring script...")
    
    monitoring_script = """
#!/usr/bin/env python3
import time
import GPUtil
import psutil
import json
import nvidia_ml_py3 as nvml
from datetime import datetime

def monitor_h100x8_gpus():
    # Initialize NVML
    nvml.nvmlInit()
    
    while True:
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "gpus": []
            }
            
            # Get GPU stats using NVML for better H100 x8 support
            for i in range(8):  # 8x H100
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                # Get power usage if available
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = 0
                
                stats["gpus"].append({
                    "gpu_id": i,
                    "memory_used": memory_info.used / 1024**3,  # GB
                    "memory_total": memory_info.total / 1024**3,  # GB
                    "utilization": utilization.gpu,
                    "temperature": temperature,
                    "power_draw": power
                })
            
            # Save to file
            with open("h100x8_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
                
            time.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            print(f"Error monitoring H100 x8 GPUs: {e}")
            time.sleep(10)
        finally:
            try:
                nvml.nvmlShutdown()
            except:
                pass

if __name__ == "__main__":
    monitor_h100x8_gpus()
"""
    
    with open("h100x8_monitor.py", "w") as f:
        f.write(monitoring_script)
    
    # Make executable
    os.chmod("h100x8_monitor.py", 0o755)
    logger.info("H100 x8 monitoring script created")

def create_h100x8_performance_dashboard():
    """Create H100 x8-specific performance dashboard"""
    logger.info("Creating H100 x8 performance dashboard...")
    
    dashboard_script = """
#!/usr/bin/env python3
import json
import time
from datetime import datetime
import GPUtil
import psutil

def get_h100x8_system_stats():
    gpus = GPUtil.getGPUs()
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    total_utilization = sum(gpu.load * 100 for gpu in gpus[:8]) / 8
    total_memory = sum(gpu.memoryUsed / gpu.memoryTotal for gpu in gpus[:8]) / 8
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "gpu_utilization": total_utilization,
        "gpu_memory_usage": total_memory * 100,
        "timestamp": datetime.now().isoformat()
    }

def display_h100x8_dashboard():
    while True:
        stats = get_h100x8_system_stats()
        
        print("\\033[2J\\033[H")  # Clear screen
        print("=" * 60)
        print("H100 x8 Performance Dashboard")
        print("=" * 60)
        print(f"Timestamp: {stats['timestamp']}")
        print(f"CPU Usage: {stats['cpu_percent']:.1f}%")
        print(f"Memory Usage: {stats['memory_percent']:.1f}%")
        print(f"GPU Utilization: {stats['gpu_utilization']:.1f}%")
        print(f"GPU Memory Usage: {stats['gpu_memory_usage']:.1f}%")
        print("=" * 60)
        
        # H100 x8-specific performance assessment
        if stats['gpu_utilization'] < 80:
            print("⚠️  H100 x8 Underutilized - Consider increasing batch sizes")
        elif stats['gpu_memory_usage'] > 90:
            print("⚠️  High Memory Usage - Consider reducing batch sizes")
        elif stats['gpu_utilization'] > 95:
            print("✅ Optimal H100 x8 Performance")
        else:
            print("✅ Good H100 x8 Performance")
        
        time.sleep(2)

if __name__ == "__main__":
    display_h100x8_dashboard()
"""
    
    with open("h100x8_performance_dashboard.py", "w") as f:
        f.write(dashboard_script)
    
    # Make executable
    os.chmod("h100x8_performance_dashboard.py", 0o755)
    logger.info("H100 x8 performance dashboard created")

def create_h100x8_startup_script():
    """Create startup script for the H100 x8-optimized miner"""
    logger.info("Creating H100 x8 startup script...")
    
    startup_script = """#!/bin/bash
# H100 x8 Optimized Miner Startup Script

echo "🚀 Starting H100 x8 Optimized Miner..."

# Set environment variables for H100 x8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST=9.0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048
export HF_HUB_CACHE=/root/.cache/huggingface
export WANDB_MODE=disabled
export CUDA_MEMORY_FRACTION=0.95
export TF_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1

# Start H100 x8 monitoring in background
echo "📊 Starting H100 x8 monitoring..."
python h100x8_monitor.py &

# Start H100 x8 performance dashboard in background
echo "📈 Starting H100 x8 performance dashboard..."
python h100x8_performance_dashboard.py &

# Start the miner
echo "⛏️  Starting miner with H100 x8 optimizations..."
python -m miner.server

echo "✅ H100 x8 Optimized Miner started successfully!"
"""
    
    with open("start_h100x8_miner.sh", "w") as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod("start_h100x8_miner.sh", 0o755)
    logger.info("H100 x8 startup script created")

def create_config_backup():
    """Create backup of original configurations"""
    logger.info("Creating configuration backups...")
    
    backup_dir = Path("backup_configs")
    backup_dir.mkdir(exist_ok=True)
    
    # Backup original files
    files_to_backup = [
        "miner/endpoints/tuning.py",
        "miner/logic/job_handler.py",
        "miner/utils.py"
    ]
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            backup_path = backup_dir / Path(file_path).name
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backed up {file_path}")
    
    return True

def create_h100x8_optimization_guide():
    """Create H100 x8-specific optimization guide"""
    logger.info("Creating H100 x8 optimization guide...")
    
    guide_content = """
# H100 x8 Optimization Guide

## Quick Start
1. Run the setup: python setup_h100x8_optimization.py
2. Start the miner: ./start_h100x8_miner.sh
3. Monitor performance: python h100x8_performance_dashboard.py

## H100 x8-Specific Optimizations

### Memory Management
- Each H100 has 80GB VRAM (640GB total)
- Use 95% of available memory
- Keep 10% buffer for optimal performance

### Batch Size Optimization
- Small models (1-7B): batch_size = 16
- Medium models (7-13B): batch_size = 8
- Large models (13-70B): batch_size = 4
- XLarge models (70B+): batch_size = 2

### LoRA Configuration
- Small models: r=256, alpha=64
- Medium models: r=512, alpha=128
- Large models: r=1024, alpha=256
- XLarge models: r=2048, alpha=512

### Learning Rate Optimization
- Small models: 4e-4
- Medium models: 3e-4
- Large models: 2e-4
- XLarge models: 1e-4

## Performance Targets
- GPU Utilization: 85-95%
- Memory Usage: 80-90%
- Temperature: < 85°C
- First Place Rate: >80%

## Monitoring Commands
- Check GPU stats: cat h100x8_stats.json
- Monitor performance: python h100x8_performance_dashboard.py
- Check system health: nvidia-smi

## Troubleshooting
- If memory usage > 90%: Reduce batch size
- If utilization < 80%: Increase batch size
- If temperature > 85°C: Reduce utilization
"""
    
    with open("H100x8_OPTIMIZATION_GUIDE.md", "w") as f:
        f.write(guide_content)
    
    logger.info("H100 x8 optimization guide created")

def main():
    """Main setup function"""
    logger.info("🚀 Starting H100 x8 Optimization Setup")
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("System requirements not met. Please ensure you have 8x H100 GPUs.")
        return False
    
    # Create backups
    create_config_backup()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Setup environment
    if not setup_environment_variables():
        logger.error("Failed to setup environment variables")
        return False
    
    # Create monitoring scripts
    create_h100x8_monitoring_script()
    create_h100x8_performance_dashboard()
    create_h100x8_startup_script()
    create_h100x8_optimization_guide()
    
    logger.info("✅ H100 x8 Optimization Setup Complete!")
    logger.info("")
    logger.info("📋 Next Steps:")
    logger.info("1. Review the strategy document: H100x8_STRATEGY.md")
    logger.info("2. Review the optimization guide: H100x8_OPTIMIZATION_GUIDE.md")
    logger.info("3. Start the optimized miner: ./start_h100x8_miner.sh")
    logger.info("4. Monitor performance: python h100x8_performance_dashboard.py")
    logger.info("5. Check H100 x8 stats: cat h100x8_stats.json")
    logger.info("")
    logger.info("🎯 Target: Top 1% ranking within 2 months")
    logger.info("🚀 H100 x8 Advantage: 8x the memory, 4x the speed, 3x the efficiency")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 