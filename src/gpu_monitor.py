"""
GPU Memory Monitor for Colab Training
Provides real-time VRAM usage tracking during fine-tuning
"""
import torch
import subprocess
import time
from datetime import datetime


def show_gpu_info():
    """Display current GPU info"""
    print(f"{'='*60}")
    print(f"GPU Monitoring - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    gpu_name = torch.cuda.get_device_name(0)
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_memory_gb:.1f} GB")
    print(f"Allocated: {allocated_gb:.2f} GB ({100*allocated_gb/total_memory_gb:.1f}%)")
    print(f"Reserved: {reserved_gb:.2f} GB ({100*reserved_gb/total_memory_gb:.1f}%)")
    print(f"Available: {total_memory_gb - reserved_gb:.2f} GB")
    print(f"{'='*60}\n")


def nvidia_smi():
    """Show nvidia-smi output"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        print(result.stdout)
    except Exception as e:
        print(f"nvidia-smi not available: {e}")


def monitor_during_training(callback_interval=100):
    """
    Returns a callback function for training loops
    
    Usage in training:
        monitor = monitor_during_training(callback_interval=100)
        # In training loop:
        if step % callback_interval == 0:
            monitor(step)
    """
    def callback(step=None):
        step_str = f"Step {step}" if step else "Monitor"
        print(f"\n{step_str}:")
        show_gpu_info()
    
    return callback


if __name__ == "__main__":
    # Standalone usage in Colab cell
    show_gpu_info()
    nvidia_smi()
