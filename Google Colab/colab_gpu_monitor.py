"""
Complete GPU & System Information Suite for Google Colab
Run this in a Google Colab notebook cell
"""

# ============================================================================
# PART 1: COMPREHENSIVE SYSTEM INFORMATION
# ============================================================================

def comprehensive_gpu_system_info():
    """Get all GPU and system information"""
    import subprocess
    import platform
    
    def run_cmd(cmd):
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "N/A"
    
    print("="*80)
    print("  COMPREHENSIVE GPU & SYSTEM INFORMATION")
    print("="*80)
    
    # Install required packages
    print("\n🔧 Installing required packages...")
    subprocess.run("pip install -q py3nvml psutil gputil", shell=True)
    
    # GPU Information
    print("\n" + "="*80)
    print("  GPU HARDWARE SPECIFICATIONS")
    print("="*80)
    
    gpu_info = run_cmd("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,pstate --format=csv,noheader")
    if gpu_info:
        parts = gpu_info.split(', ')
        print(f"\n🎮 GPU Model: {parts[0] if len(parts) > 0 else 'N/A'}")
        print(f"💾 Total Memory: {parts[1] if len(parts) > 1 else 'N/A'}")
        print(f"🔧 Driver Version: {parts[2] if len(parts) > 2 else 'N/A'}")
        print(f"⚡ Compute Capability: {parts[3] if len(parts) > 3 else 'N/A'}")
        print(f"📊 Performance State: {parts[4] if len(parts) > 4 else 'N/A'}")
    
    # Detailed GPU specs via NVML
    try:
        import py3nvml.py3nvml as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        
        print(f"\n📋 UUID: {nvml.nvmlDeviceGetUUID(handle)}")
        
        pci = nvml.nvmlDeviceGetPciInfo(handle)
        print(f"🔌 PCI Bus ID: {pci.busId}")
        
        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"\n💾 Memory Details:")
        print(f"   Total: {mem_info.total / 1024**3:.2f} GB")
        print(f"   Free: {mem_info.free / 1024**3:.2f} GB") 
        print(f"   Used: {mem_info.used / 1024**3:.2f} GB")
        
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"\n📊 Current Utilization:")
        print(f"   GPU: {util.gpu}%")
        print(f"   Memory: {util.memory}%")
        
        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        print(f"🌡️  Temperature: {temp}°C")
        
        try:
            power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000
            power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
            print(f"\n⚡ Power:")
            print(f"   Current: {power:.2f} W")
            print(f"   Limit: {power_limit:.2f} W")
        except:
            pass
        
        print(f"\n⏱️  Clock Speeds:")
        print(f"   Graphics: {nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)} MHz")
        print(f"   SM: {nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)} MHz")
        print(f"   Memory: {nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)} MHz")
        
        print(f"\n🚀 Max Clock Speeds:")
        print(f"   Graphics: {nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)} MHz")
        print(f"   SM: {nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_SM)} MHz")
        print(f"   Memory: {nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_MEM)} MHz")
        
        nvml.nvmlShutdown()
    except Exception as e:
        print(f"⚠️  Could not get detailed GPU info: {e}")
    
    # CUDA Information
    print("\n" + "="*80)
    print("  CUDA & DEEP LEARNING FRAMEWORKS")
    print("="*80)
    
    cuda_version = run_cmd("nvidia-smi | grep 'CUDA Version' | awk '{print $9}'")
    print(f"\n🔷 CUDA Version (Driver): {cuda_version}")
    
    # PyTorch
    try:
        import torch
        print(f"\n🔥 PyTorch:")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   Device Count: {torch.cuda.device_count()}")
            print(f"   Device Name: {torch.cuda.get_device_name(0)}")
            print(f"   Capability: {torch.cuda.get_device_capability(0)}")
    except ImportError:
        print("\n🔥 PyTorch: Not installed")
    
    # TensorFlow
    try:
        import tensorflow as tf
        print(f"\n🧠 TensorFlow:")
        print(f"   Version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   GPU Devices: {len(gpus)}")
        if gpus:
            print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
    except ImportError:
        print("\n🧠 TensorFlow: Not installed")
    
    # JAX
    try:
        import jax
        print(f"\n🎯 JAX:")
        print(f"   Version: {jax.__version__}")
        print(f"   Devices: {jax.devices()}")
    except ImportError:
        print("\n🎯 JAX: Not installed")
    
    # System Information
    print("\n" + "="*80)
    print("  SYSTEM SPECIFICATIONS")
    print("="*80)
    
    import psutil
    
    print(f"\n🖥️  Platform: {platform.platform()}")
    print(f"💻 System: {platform.system()} {platform.release()}")
    
    cpu_info = run_cmd("cat /proc/cpuinfo | grep 'model name' | uniq | cut -d':' -f2")
    print(f"\n🔲 CPU: {cpu_info.strip()}")
    print(f"   Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"   Logical Cores: {psutil.cpu_count(logical=True)}")
    
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        print(f"   Current Frequency: {cpu_freq.current:.2f} MHz")
        print(f"   Max Frequency: {cpu_freq.max:.2f} MHz")
    
    mem = psutil.virtual_memory()
    print(f"\n💾 System Memory:")
    print(f"   Total: {mem.total / 1024**3:.2f} GB")
    print(f"   Available: {mem.available / 1024**3:.2f} GB")
    print(f"   Used: {mem.used / 1024**3:.2f} GB ({mem.percent}%)")
    
    disk = psutil.disk_usage('/')
    print(f"\n💿 Disk:")
    print(f"   Total: {disk.total / 1024**3:.2f} GB")
    print(f"   Free: {disk.free / 1024**3:.2f} GB")
    print(f"   Used: {disk.used / 1024**3:.2f} GB ({disk.percent}%)")
    
    # GPU Processes
    print("\n" + "="*80)
    print("  ACTIVE GPU PROCESSES")
    print("="*80)
    
    processes = run_cmd("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
    print(f"\n{processes}")
    
    # Colab Environment Info
    print("\n" + "="*80)
    print("  GOOGLE COLAB ENVIRONMENT")
    print("="*80)
    
    try:
        import google.colab
        print("\n✅ Running in Google Colab")
        
        # Check for TPU
        try:
            import tensorflow as tf
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print(f"🎯 TPU Available: Yes")
        except:
            print(f"🎯 TPU Available: No")
        
        # Mounted drives
        import os
        if os.path.exists('/content/drive'):
            print("📁 Google Drive: Mounted")
        else:
            print("📁 Google Drive: Not mounted")
            
    except:
        print("\n⚠️  Not running in Google Colab")
    
    print("\n" + "="*80)
    print("  REPORT COMPLETE")
    print("="*80 + "\n")


# ============================================================================
# PART 2: REAL-TIME MONITORING DASHBOARD
# ============================================================================

def start_monitoring(duration=60, interval=1):
    """Start real-time GPU monitoring with visualizations"""
    import time
    import subprocess
    import matplotlib.pyplot as plt
    from IPython import display
    import numpy as np
    from datetime import datetime
    
    print(f"\n🚀 Starting real-time GPU monitoring...")
    print(f"Duration: {duration} seconds | Update interval: {interval} second(s)")
    print("="*80 + "\n")
    
    history = {
        'time': [],
        'gpu_util': [],
        'mem_used': [],
        'mem_total': [],
        'temp': [],
        'power': []
    }
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Get metrics
            cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                
                history['time'].append(time.time() - start_time)
                history['gpu_util'].append(float(values[0]))
                history['mem_used'].append(float(values[1]))
                history['mem_total'].append(float(values[2]))
                history['temp'].append(float(values[3]))
                history['power'].append(float(values[4]) if values[4] != '[N/A]' else 0)
                
                # Keep last 100 points
                for key in history:
                    if len(history[key]) > 100:
                        history[key] = history[key][-100:]
                
                # Plot
                display.clear_output(wait=True)
                
                fig, axes = plt.subplots(2, 3, figsize=(16, 10))
                fig.suptitle(f'GPU Real-Time Monitor - {datetime.now().strftime("%H:%M:%S")}', 
                           fontsize=14, fontweight='bold')
                
                # GPU Utilization
                axes[0, 0].plot(history['time'], history['gpu_util'], 'b-', linewidth=2)
                axes[0, 0].fill_between(history['time'], history['gpu_util'], alpha=0.3)
                axes[0, 0].set_ylabel('GPU Utilization (%)')
                axes[0, 0].set_title('GPU Usage')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_ylim(0, 100)
                
                # Memory
                axes[0, 1].plot(history['time'], history['mem_used'], 'r-', linewidth=2, label='Used')
                axes[0, 1].axhline(y=history['mem_total'][-1], color='g', linestyle='--', label='Total')
                axes[0, 1].fill_between(history['time'], history['mem_used'], alpha=0.3)
                axes[0, 1].set_ylabel('Memory (MB)')
                axes[0, 1].set_title('GPU Memory')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Temperature
                axes[0, 2].plot(history['time'], history['temp'], 'orange', linewidth=2)
                axes[0, 2].fill_between(history['time'], history['temp'], alpha=0.3, color='orange')
                axes[0, 2].set_ylabel('Temperature (°C)')
                axes[0, 2].set_title('GPU Temperature')
                axes[0, 2].grid(True, alpha=0.3)
                
                # Power
                if any(p > 0 for p in history['power']):
                    axes[1, 0].plot(history['time'], history['power'], 'purple', linewidth=2)
                    axes[1, 0].fill_between(history['time'], history['power'], alpha=0.3, color='purple')
                    axes[1, 0].set_ylabel('Power (W)')
                    axes[1, 0].set_title('Power Draw')
                    axes[1, 0].set_xlabel('Time (seconds)')
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    axes[1, 0].text(0.5, 0.5, 'Power data not available', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                
                # Memory %
                mem_pct = [(u/t*100) if t > 0 else 0 for u, t in zip(history['mem_used'], history['mem_total'])]
                axes[1, 1].plot(history['time'], mem_pct, 'brown', linewidth=2)
                axes[1, 1].fill_between(history['time'], mem_pct, alpha=0.3, color='brown')
                axes[1, 1].set_ylabel('Utilization (%)')
                axes[1, 1].set_title('Memory Utilization %')
                axes[1, 1].set_xlabel('Time (seconds)')
                axes[1, 1].set_ylim(0, 100)
                axes[1, 1].grid(True, alpha=0.3)
                
                # Stats
                axes[1, 2].axis('off')
                stats = f"""CURRENT STATS
                
GPU Util: {history['gpu_util'][-1]:.1f}%
Avg: {np.mean(history['gpu_util']):.1f}%
Max: {np.max(history['gpu_util']):.1f}%

Memory: {history['mem_used'][-1]:.0f} / {history['mem_total'][-1]:.0f} MB
Usage: {mem_pct[-1]:.1f}%

Temperature: {history['temp'][-1]:.1f}°C
Avg: {np.mean(history['temp']):.1f}°C
Max: {np.max(history['temp']):.1f}°C

Time: {history['time'][-1]:.1f}s / {duration}s
"""
                axes[1, 2].text(0.1, 0.9, stats, transform=axes[1, 2].transAxes,
                              fontsize=11, verticalalignment='top', family='monospace',
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                
                plt.tight_layout()
                plt.show()
                
                # Console output
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                print(f"⏱️  {elapsed:.1f}s elapsed | {remaining:.1f}s remaining")
                print(f"📊 GPU: {history['gpu_util'][-1]:.1f}% | "
                      f"Mem: {history['mem_used'][-1]:.0f}MB ({mem_pct[-1]:.1f}%) | "
                      f"Temp: {history['temp'][-1]:.1f}°C | "
                      f"Power: {history['power'][-1]:.1f}W")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n⚠️  Monitoring stopped by user")
    
    print(f"\n✅ Monitoring complete! Total time: {time.time() - start_time:.1f}s")
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Part 1: Show comprehensive system info
    comprehensive_gpu_system_info()
    
    # Part 2: Ask if user wants real-time monitoring
    print("\n" + "="*80)
    response = input("Start real-time monitoring? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            duration = int(input("Duration in seconds (default 60): ") or "60")
            interval = float(input("Update interval in seconds (default 1): ") or "1")
            start_monitoring(duration=duration, interval=interval)
        except:
            print("Invalid input, using defaults...")
            start_monitoring(duration=60, interval=1)
    else:
        print("\n✅ System information gathering complete!")
        print("="*80 + "\n")
