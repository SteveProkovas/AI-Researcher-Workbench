# 🔍 GPU & System Monitoring Suite for Google Colab

**Professional monitoring tools for NVIDIA T4 GPU performance on Google Colab with real-time dashboards, system diagnostics, and performance analytics.**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab Compatible](https://img.shields.io/badge/Colab-Compatible-brightgreen)](https://colab.research.google.com)
[![NVIDIA T4](https://img.shields.io/badge/GPU-NVIDIA_T4-76b900)](https://www.nvidia.com/en-us/data-center/tesla-t4/)

## 📋 Table of Contents
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [Monitoring Dashboard](#-monitoring-dashboard)
- [Understanding Metrics](#-understanding-metrics)
- [Use Cases](#-use-cases)
- [Troubleshooting](#-troubleshooting)
- [Colab Best Practices](#-colab-best-practices)
- [API Reference](#-api-reference)
- [FAQ](#-faq)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 📊 **Comprehensive Monitoring**
| Category | Metrics | Details |
|----------|---------|---------|
| **GPU Hardware** | Model, Memory, UUID | T4 specifications, 15-16GB VRAM, PCI bus info |
| **Performance** | Utilization, Temperature | Real-time usage, thermal monitoring (60-85°C range) |
| **Power & Clocks** | Power draw, Clock speeds | 70W TDP, SM/Memory frequencies |
| **System Info** | CPU, RAM, Disk | CPU cores, memory usage, storage space |
| **DL Frameworks** | PyTorch, TensorFlow, JAX | Version compatibility, CUDA/cuDNN support |

### 🎯 **Three Monitoring Modes**

| File | Purpose | Best For | Visualization |
|------|---------|----------|---------------|
| `colab_gpu_monitor.py` | **All-in-one solution** | General users, quick diagnostics | ✅ Real-time graphs |
| `gpu_system_info.py` | **Static system info** | Fast checks, no monitoring | ❌ Text only |
| `gpu_dashboard.py` | **Advanced dashboard** | Long-term monitoring, research | ✅ 9 detailed graphs |

## 🚀 Quick Start

### **Option 1: One-Line Installation (Recommended)**
```python
# Run in a Colab cell
!wget -q https://raw.githubusercontent.com/your-username/your-repo/main/colab_gpu_monitor.py
%run colab_gpu_monitor.py
```

### **Option 2: Manual Upload**
1. Download `colab_gpu_monitor.py` from this repository
2. Upload to Colab via Files sidebar
3. Run:
```python
%run colab_gpu_monitor.py
```

### **Option 3: Direct Execution**
Copy the script content into a Colab cell and execute.

## 📦 Installation

### **Dependencies**
All packages are auto-installed:
```python
# Core monitoring
py3nvml    # NVIDIA Management Library
psutil     # System process utilities
gputil     # GPU utilities

# Visualization
matplotlib # Plotting library
numpy      # Numerical operations

# Built-in Python libraries
import subprocess, platform, time, datetime, threading, json
```

### **Verification**
```python
# Verify GPU access
from colab_gpu_monitor import check_gpu_availability
check_gpu_availability()  # Returns (True, "Tesla T4") or error message
```

## 💻 Usage Examples

### **1. Quick System Snapshot**
```python
from colab_gpu_monitor import get_system_snapshot

snapshot = get_system_snapshot()
print(f"GPU: {snapshot['gpu']['name']}")
print(f"Memory: {snapshot['gpu']['memory_used']}/{snapshot['gpu']['memory_total']} GB")
print(f"CUDA: {snapshot['cuda']['version']}")
```

### **2. Real-Time Monitoring**
```python
from colab_gpu_monitor import start_monitoring

# Monitor for 5 minutes with 2-second intervals
start_monitoring(duration=300, interval=2)

# Or with custom callbacks
def on_high_temp(temp, threshold=80):
    if temp > threshold:
        print(f"⚠️ Warning: High temperature ({temp}°C)")

start_monitoring(duration=600, interval=1, callbacks={'temperature': on_high_temp})
```

### **3. Advanced Dashboard**
```python
from gpu_dashboard import GPUMonitor

# Customizable monitoring
monitor = GPUMonitor(
    max_history=200,      # Store 200 data points
    save_logs=True,       # Save metrics to file
    alert_thresholds={    # Custom alerts
        'temperature': 80,
        'memory_usage': 0.9  # 90%
    }
)

# Start monitoring
monitor.start(duration=1200)  # 20 minutes

# Export data
monitor.export_csv('gpu_metrics.csv')
monitor.save_plots('monitoring_report.png')
```

### **4. Integration with Model Training**
```python
import threading
from colab_gpu_monitor import BackgroundMonitor

# Start monitoring in background
monitor = BackgroundMonitor(interval=5)  # Check every 5 seconds
monitor.start()

# Your training code
import torch
model = torch.nn.Linear(10, 10)
# ... training loop ...

# Stop monitoring and get report
report = monitor.stop()
print(f"Peak memory: {report['peak_memory_gb']:.2f} GB")
```

## 📊 Monitoring Dashboard

The advanced dashboard displays 9 real-time graphs:

| Graph | Metric | Normal Range | Alert Threshold |
|-------|--------|--------------|-----------------|
| **GPU Utilization** | % of GPU cores active | 30-100% | <20% (underutilized) |
| **Memory Usage** | VRAM consumption (GB) | 0-15 GB | >14 GB (near limit) |
| **Temperature** | GPU core temperature | 60-80°C | >83°C (throttling) |
| **Power Draw** | Electrical power (W) | 40-70W | >70W (at TDP limit) |
| **SM Clock** | Streaming multiprocessor (MHz) | 585-1590MHz | - |
| **Memory Clock** | VRAM frequency (MHz) | 3000-5000MHz | - |
| **Memory Utilization** | % of VRAM used | 0-100% | >95% (risk of OOM) |
| **Process Count** | Active GPU processes | 1-10 | >15 (potentially shared) |
| **Fan Speed** | Cooling fan (% of max) | 0-100% | >90% (high cooling) |

## 🎯 Understanding Metrics

### **GPU Utilization Guide**
| Range | Interpretation | Action |
|-------|----------------|--------|
| **0-30%** | Underutilized | Increase batch size, enable mixed precision |
| **30-70%** | Good utilization | Optimal for most workflows |
| **70-100%** | High utilization | Monitor temperature, may be bottlenecked |

### **Memory Management**
```python
# Calculate safe batch size
def calculate_max_batch_size(model_memory_gb, available_vram_gb=15):
    safety_margin = 2.0  # GB for overhead
    max_batch_memory = available_vram_gb - model_memory_gb - safety_margin
    return int(max_batch_memory * 1000)  # Approximate elements
```

### **Thermal Zones**
- **< 65°C**: ✅ Optimal (green zone)
- **65-80°C**: ⚠️ Acceptable (yellow zone)
- **> 80°C**: 🔥 Critical (red zone, throttling possible)

## 🔧 Use Cases

### **1. Model Training Optimization**
```python
# Monitor during hyperparameter tuning
from colab_gpu_monitor import TrainingMonitor

monitor = TrainingMonitor()
monitor.start()

for batch_size in [16, 32, 64, 128]:
    print(f"Testing batch size: {batch_size}")
    # Run training epoch
    metrics = monitor.capture_epoch()
    print(f"GPU Util: {metrics['avg_util']}%, Memory: {metrics['peak_memory']}GB")
```

### **2. Performance Benchmarking**
```python
# Compare different operations
benchmark_results = {}
operations = ['matrix_mult', 'conv2d', 'attention']

for op in operations:
    monitor.clear_history()
    # Execute operation
    run_operation(op)
    stats = monitor.get_statistics()
    benchmark_results[op] = stats
```

### **3. Resource Leak Detection**
```python
# Monitor for memory leaks
from colab_gpu_monitor import MemoryLeakDetector

detector = MemoryLeakDetector(monitoring_interval=2)
leak_found = detector.monitor_session(duration=300)

if leak_found:
    print("⚠️ Potential memory leak detected!")
    detector.identify_suspicious_processes()
```

## 🐛 Troubleshooting

### **Common Issues & Solutions**

| Issue | Solution | Code Fix |
|-------|----------|----------|
| **No GPU detected** | Enable GPU in Colab runtime | Runtime → Change runtime type → GPU |
| **Module not found** | Install missing packages | `!pip install py3nvml psutil gputil matplotlib` |
| **Permission errors** | Update NVML permissions | `!sudo chmod 644 /proc/driver/nvidia/gpus/*/information` |
| **Graphs not updating** | Clear output and rerun | `from IPython.display import clear_output; clear_output()` |
| **Monitoring stops** | Colab timeout (90min idle) | Use audio alert: `output.eval_js('new Audio(...).play()')` |

### **Diagnostic Commands**
```python
# Run diagnostic suite
from colab_gpu_monitor import run_diagnostics

diagnostics = run_diagnostics()
if diagnostics['all_passed']:
    print("✅ All systems operational")
else:
    for issue in diagnostics['issues']:
        print(f"❌ {issue}")
```

## 📝 Colab Best Practices

### **Session Management**
```python
# Save monitoring data before session ends
import pickle
from google.colab import files

def save_session_data(monitor):
    data = monitor.get_historical_data()
    with open('gpu_session.pkl', 'wb') as f:
        pickle.dump(data, f)
    files.download('gpu_session.pkl')  # Auto-download
```

### **Pro/Pro+ Tips**
- **Longer sessions**: Up to 24 hours on Pro+
- **Better GPUs**: Occasional V100/A100 access
- **Priority access**: Reduced queue times
- **Higher memory**: Sometimes 25GB+ instances

### **Resource Limits**
| Resource | Free Tier | Pro | Pro+ |
|----------|-----------|-----|------|
| **Session timeout** | 12 hours | 24 hours | 24 hours |
| **Idle disconnect** | 90 minutes | 90 minutes | 90 minutes |
| **GPU availability** | Not guaranteed | Higher priority | Highest priority |
| **Memory** | ~12GB RAM | ~32GB RAM | ~52GB RAM |

## 📚 API Reference

### **Core Functions**
```python
# System Information
get_gpu_details()          # GPU specifications
get_system_info()          # CPU, RAM, OS details
get_cuda_info()            # CUDA and framework versions

# Monitoring
start_monitoring(duration, interval)  # Basic monitoring
create_dashboard()         # Interactive dashboard
background_monitor()       # Non-blocking monitoring

# Utilities
export_metrics(format='csv')  # Export data
generate_report()          # HTML/PDF report
set_alerts(thresholds)     # Custom alerts
```

### **Configuration**
```python
# Custom configuration
config = {
    'update_interval': 1.0,      # Seconds between updates
    'log_to_file': True,         # Save metrics to file
    'max_data_points': 1000,     # History limit
    'visualization_engine': 'matplotlib',  # or 'plotly'
    'auto_optimize': True        # Suggest optimizations
}
```

## ❓ FAQ

### **Q: Is this tool free to use?**
**A:** Yes, completely free and open-source under MIT license.

### **Q: Does it work with other GPUs besides T4?**
**A:** Yes! Works with any NVIDIA GPU (P100, V100, A100, etc.), but metrics are optimized for T4.

### **Q: Will monitoring affect my GPU performance?**
**A:** Minimal impact (<1% GPU utilization). Designed to be lightweight.

### **Q: Can I use this outside Colab?**
**A:** Yes, works on any system with NVIDIA GPU and Python 3.7+.

### **Q: How do I save monitoring data?**
**A:** Use the export functions:
```python
monitor.export_csv('data.csv')
monitor.save_plots('report.png')
monitor.generate_html_report('dashboard.html')
```

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. **Report bugs** via GitHub Issues
2. **Suggest features** for future versions
3. **Submit pull requests** with improvements
4. **Share use cases** and success stories

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/your-username/gpu-monitor-colab.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black *.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Credits & Acknowledgments

- **NVIDIA** for NVML and management libraries
- **Google Colab** for providing GPU resources
- **Open-source contributors** of dependent packages
- **Turing Architecture** documentation

### **Related Projects**
- [NVIDIA ML Perf](https://github.com/mlperf) - Performance benchmarks
- [Weights & Biases](https://wandb.ai) - Experiment tracking
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - Detailed profiling

---

**📅 Last Updated:** January 2026  
**🔧 Maintained by:** Stavros Steve Prokovas  

*For questions, issues, or contributions, please open a GitHub Issue or Discussion.*
