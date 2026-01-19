# GPU & System Monitoring Suite for Google Colab
## Complete toolkit for monitoring NVIDIA T4 GPU performance on Google Colab with detailed system information and real-time visualizations.
📦 Files Included

colab_gpu_monitor.py - ⭐ RECOMMENDED - All-in-one solution

Comprehensive system information
Real-time monitoring dashboard
Interactive prompts
Best for general use


gpu_system_info.py - Detailed static information

Complete GPU specifications
System details
CUDA/framework versions
No visualization (fastest)


gpu_dashboard.py - Advanced real-time monitoring

Professional dashboard with 9 graphs
Detailed statistics
Customizable duration
Best for long-term monitoring



## 🚀 Quick Start (Google Colab)
Option 1: Simple One-Liner (Recommended)
python# Run this in a Colab cell
!wget -q https://raw.githubusercontent.com/YOUR_REPO/colab_gpu_monitor.py
%run colab_gpu_monitor.py
Option 2: Copy-Paste Method

Download colab_gpu_monitor.py from this repository
Upload to your Colab session
Run:

python%run colab_gpu_monitor.py
Option 3: Direct Code Execution
Copy the entire contents of colab_gpu_monitor.py into a Colab cell and run it.
📊 What You'll Get
System Information Includes:
✅ GPU Hardware:

Model name (Tesla T4)
Total memory (typically 15-16 GB)
Driver version
Compute capability
PCI bus information
UUID
Clock speeds (current and max)
Power limits

✅ Performance Metrics:

Real-time GPU utilization
Memory usage (used/free/total)
Temperature monitoring
Power consumption
SM and memory clock frequencies

✅ CUDA & Deep Learning:

CUDA version
PyTorch compatibility and version
TensorFlow compatibility and version
JAX support (if installed)
cuDNN version

✅ System Specifications:

CPU model and cores
RAM capacity and usage
Disk space
Operating system details

✅ Active Processes:

GPU process list
Memory per process
Process IDs

Real-Time Dashboard Features:
📈 Live Graphs:

GPU Utilization (%)
Memory Usage (MB)
Temperature (°C)
Power Draw (W)
SM Clock Speed (MHz)
Memory Clock Speed (MHz)
Memory Utilization (%)
Statistics Summary
GPU Information Panel

💻 Usage Examples
Example 1: Quick System Check
python# Just get system information
from colab_gpu_monitor import comprehensive_gpu_system_info
comprehensive_gpu_system_info()
Example 2: Monitor for 2 Minutes
python# Monitor GPU for 2 minutes with 1-second updates
from colab_gpu_monitor import start_monitoring
start_monitoring(duration=120, interval=1)
Example 3: Long-Term Monitoring
python# Monitor for 10 minutes with 2-second updates
from colab_gpu_monitor import start_monitoring
start_monitoring(duration=600, interval=2)
Example 4: Use Advanced Dashboard
pythonfrom gpu_dashboard import GPUMonitor

## Create monitor instance
monitor = GPUMonitor(max_history=150)  # Keep 150 data points

## Start monitoring for 5 minutes
monitor.monitor(duration=300, interval=1)
🎯 Use Cases
1. Model Training Monitoring
Monitor GPU usage while training your AI models to ensure efficient resource utilization.
python# Start monitoring in one cell
import threading
from colab_gpu_monitor import start_monitoring

monitor_thread = threading.Thread(
    target=start_monitoring, 
    args=(3600, 2)  # 1 hour, 2-second updates
)
monitor_thread.start()

## Train your model in another cell
## Your training code here...
2. Performance Benchmarking
Compare GPU performance across different model architectures or batch sizes.
3. Resource Optimization
Identify bottlenecks and optimize your code for better GPU utilization.
4. Thermal Monitoring
Ensure your GPU doesn't overheat during intensive operations.
5. Multi-User Detection
Check if GPU resources are being shared (though Colab typically provides dedicated instances).
⚙️ Requirements
All required packages are automatically installed:

py3nvml - NVIDIA Management Library Python bindings
psutil - System and process utilities
gputil - GPU utilities
matplotlib - Plotting library
Standard libraries: subprocess, platform, time, datetime

🔍 Understanding the Metrics
GPU Utilization

0-30%: Underutilized - Consider increasing batch size
30-70%: Good utilization
70-100%: Excellent utilization - GPU is working hard

Memory Usage

Monitor to avoid OOM (Out of Memory) errors
T4 has ~15GB VRAM
Leave headroom for peaks

Temperature

< 65°C: Cool (Green zone)
65-80°C: Warm (Yellow zone)
> 80°C: Hot (Red zone) - Usually throttles above 83°C

Power Draw

T4 TDP is 70W
Should stay below this limit
Higher = more computation happening

🐛 Troubleshooting
"No GPU detected"
python# Check GPU is enabled in Colab
## Runtime -> Change runtime type -> Hardware accelerator -> GPU
"Module not found" errors
python# Install missing packages
!pip install py3nvml psutil gputil matplotlib
Monitoring stops unexpectedly
python# Colab may disconnect after 90 minutes of inactivity
## Keep the tab active or use:
from google.colab import output
output.eval_js('new Audio("https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3").play()')
Graphs not updating
python# Make sure you're running in a notebook cell, not a script
## Clear outputs and try again
from IPython.display import clear_output
clear_output()
📝 Notes for Google Colab

Session Limits:

Free tier: ~12 hours max session
GPU availability not guaranteed
May disconnect after 90 min idle


GPU Allocation:

Usually dedicated T4 instance
Not shared with other users on same GPU
But your VM might be on shared host


Persistence:

Monitoring data is not saved between sessions
Download graphs/logs before session ends


Pro/Pro+ Benefits:

Longer sessions
Better GPU availability
Sometimes faster GPUs (V100, A100)



🔗 Additional Resources

NVIDIA T4 Specifications
NVIDIA T4 Whitepaper
Google Colab GPU FAQ
NVIDIA Management Library

📧 Support
If you encounter issues:

Check that GPU is enabled in Colab settings
Verify all packages are installed
Try restarting the runtime
Check Colab usage quotas

📄 License
Free to use and modify for any purpose.
🙏 Credits
Created using:

NVIDIA NVML/nvidia-smi
Python libraries: py3nvml, psutil, matplotlib
Turing Architecture Whitepaper (included in your uploads)


Last Updated: January 2026
Compatible with: Google Colab, NVIDIA T4 GPU
Python Version: 3.7+
