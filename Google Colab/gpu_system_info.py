"""
Comprehensive GPU and System Information Dashboard
For Google Colab NVIDIA T4 Instance
"""

import subprocess
import platform
import psutil
import os
from datetime import datetime

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def get_gpu_info_nvidia_smi():
    """Get detailed GPU info using nvidia-smi"""
    print_section("NVIDIA GPU DETAILED INFORMATION (nvidia-smi)")
    
    # Full detailed query
    queries = [
        "name",
        "driver_version", 
        "pci.bus_id",
        "vbios_version",
        "compute_cap",
        "memory.total",
        "memory.free",
        "memory.used",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "temperature.memory",
        "power.draw",
        "power.limit",
        "power.max_limit",
        "power.min_limit",
        "clocks.current.graphics",
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.current.video",
        "clocks.max.graphics",
        "clocks.max.sm", 
        "clocks.max.memory",
        "fan.speed",
        "pstate",
        "persistence_mode",
        "ecc.mode.current",
        "encoder.stats.sessionCount",
        "encoder.stats.averageFps",
        "encoder.stats.averageLatency",
    ]
    
    query_string = ",".join(queries)
    cmd = f"nvidia-smi --query-gpu={query_string} --format=csv,noheader,nounits"
    
    output = run_command(cmd)
    if output and "Error" not in output:
        values = output.split(", ")
        for i, query in enumerate(queries):
            if i < len(values):
                print(f"{query:.<40} {values[i]}")
    
    print("\n" + "-"*80)

def get_gpu_info_pynvml():
    """Get GPU info using pynvml (Python NVML wrapper)"""
    try:
        import py3nvml.py3nvml as nvml
        
        print_section("GPU INFORMATION (via NVML)")
        
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            
            print(f"GPU {i}:")
            print(f"  Name: {nvml.nvmlDeviceGetName(handle)}")
            
            # UUID
            try:
                uuid = nvml.nvmlDeviceGetUUID(handle)
                print(f"  UUID: {uuid}")
            except:
                pass
            
            # PCI Info
            try:
                pci_info = nvml.nvmlDeviceGetPciInfo(handle)
                print(f"  PCI Bus ID: {pci_info.busId}")
                print(f"  PCI Device ID: 0x{pci_info.pciDeviceId:X}")
                print(f"  PCI Sub System ID: 0x{pci_info.pciSubSystemId:X}")
            except:
                pass
            
            # Compute Capability
            try:
                major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
                print(f"  Compute Capability: {major}.{minor}")
            except:
                pass
            
            # Memory
            try:
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"  Memory Total: {mem_info.total / 1024**3:.2f} GB")
                print(f"  Memory Free: {mem_info.free / 1024**3:.2f} GB")
                print(f"  Memory Used: {mem_info.used / 1024**3:.2f} GB")
                print(f"  Memory Utilization: {(mem_info.used/mem_info.total)*100:.1f}%")
            except:
                pass
            
            # Utilization
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                print(f"  GPU Utilization: {util.gpu}%")
                print(f"  Memory Controller Utilization: {util.memory}%")
            except:
                pass
            
            # Temperature
            try:
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                print(f"  GPU Temperature: {temp}°C")
            except:
                pass
            
            # Power
            try:
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000
                power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                print(f"  Power Usage: {power:.2f} W")
                print(f"  Power Limit: {power_limit:.2f} W")
            except:
                pass
            
            # Clock Speeds
            try:
                graphics_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
                sm_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                mem_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                print(f"  Graphics Clock: {graphics_clock} MHz")
                print(f"  SM Clock: {sm_clock} MHz")
                print(f"  Memory Clock: {mem_clock} MHz")
            except:
                pass
            
            # Max Clock Speeds
            try:
                max_graphics = nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
                max_sm = nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_SM)
                max_mem = nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_MEM)
                print(f"  Max Graphics Clock: {max_graphics} MHz")
                print(f"  Max SM Clock: {max_sm} MHz")
                print(f"  Max Memory Clock: {max_mem} MHz")
            except:
                pass
            
            # Performance State
            try:
                pstate = nvml.nvmlDeviceGetPerformanceState(handle)
                print(f"  Performance State: P{pstate}")
            except:
                pass
            
            # Fan Speed
            try:
                fan_speed = nvml.nvmlDeviceGetFanSpeed(handle)
                print(f"  Fan Speed: {fan_speed}%")
            except:
                pass
            
            # Encoder/Decoder Utilization
            try:
                enc_util, enc_sample = nvml.nvmlDeviceGetEncoderUtilization(handle)
                dec_util, dec_sample = nvml.nvmlDeviceGetDecoderUtilization(handle)
                print(f"  Encoder Utilization: {enc_util}%")
                print(f"  Decoder Utilization: {dec_util}%")
            except:
                pass
            
            # BAR1 Memory
            try:
                bar1 = nvml.nvmlDeviceGetBAR1MemoryInfo(handle)
                print(f"  BAR1 Memory Total: {bar1.bar1Total / 1024**3:.2f} GB")
                print(f"  BAR1 Memory Used: {bar1.bar1Used / 1024**3:.2f} GB")
            except:
                pass
            
            # Multi-GPU Link (NVLink)
            try:
                for link in range(nvml.NVML_NVLINK_MAX_LINKS):
                    try:
                        state = nvml.nvmlDeviceGetNvLinkState(handle, link)
                        if state:
                            print(f"  NVLink {link}: Active")
                    except:
                        pass
            except:
                pass
            
            print()
        
        nvml.nvmlShutdown()
        
    except ImportError:
        print("py3nvml not installed. Installing...")
        os.system("pip install py3nvml > /dev/null 2>&1")
        print("Please run the script again.")
    except Exception as e:
        print(f"Error getting GPU info via NVML: {e}")

def get_cuda_info():
    """Get CUDA information"""
    print_section("CUDA INFORMATION")
    
    # CUDA version from nvidia-smi
    cuda_version = run_command("nvidia-smi | grep 'CUDA Version' | awk '{print $9}'")
    print(f"CUDA Version (Driver): {cuda_version}")
    
    # NVCC version if available
    nvcc_version = run_command("nvcc --version 2>/dev/null | grep 'release' | awk '{print $5}' | tr -d ','")
    if nvcc_version and "Error" not in nvcc_version:
        print(f"NVCC Version: {nvcc_version}")
    
    # cuDNN version
    cudnn_version = run_command("cat /usr/local/cuda/include/cudnn_version.h 2>/dev/null | grep CUDNN_MAJOR -A 2")
    if cudnn_version and "Error" not in cudnn_version:
        print(f"\ncuDNN Info:\n{cudnn_version}")
    
    # Check PyTorch CUDA
    try:
        import torch
        print(f"\nPyTorch CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA Version: {torch.version.cuda}")
            print(f"PyTorch cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"Current CUDA Device: {torch.cuda.current_device()}")
                print(f"Device Name: {torch.cuda.get_device_name(0)}")
                print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    except:
        print("\nPyTorch not installed or CUDA not available")
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        print(f"\nTensorFlow GPU Devices: {len(tf.config.list_physical_devices('GPU'))}")
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print(f"TensorFlow Built with CUDA: {tf.test.is_built_with_cuda()}")
            print(f"TensorFlow GPU Available: {tf.test.is_gpu_available()}")
    except:
        print("\nTensorFlow not installed or GPU not available")

def get_system_info():
    """Get system information"""
    print_section("SYSTEM INFORMATION")
    
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()}")
    print(f"Release: {platform.release()}")
    print(f"Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # CPU Info
    print(f"\nCPU Count (Physical): {psutil.cpu_count(logical=False)}")
    print(f"CPU Count (Logical): {psutil.cpu_count(logical=True)}")
    
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        print(f"CPU Frequency - Current: {cpu_freq.current:.2f} MHz")
        print(f"CPU Frequency - Min: {cpu_freq.min:.2f} MHz")
        print(f"CPU Frequency - Max: {cpu_freq.max:.2f} MHz")
    
    # Detailed CPU model
    cpu_info = run_command("cat /proc/cpuinfo | grep 'model name' | uniq")
    if cpu_info:
        print(f"\n{cpu_info}")
    
    print(f"\nCPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"  Core {i}: {percentage}%")
    
    print(f"\nTotal CPU Usage: {psutil.cpu_percent(interval=1)}%")

def get_memory_info():
    """Get memory information"""
    print_section("SYSTEM MEMORY INFORMATION")
    
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    print(f"Total RAM: {mem.total / 1024**3:.2f} GB")
    print(f"Available RAM: {mem.available / 1024**3:.2f} GB")
    print(f"Used RAM: {mem.used / 1024**3:.2f} GB")
    print(f"RAM Usage: {mem.percent}%")
    
    print(f"\nSwap Total: {swap.total / 1024**3:.2f} GB")
    print(f"Swap Used: {swap.used / 1024**3:.2f} GB")
    print(f"Swap Free: {swap.free / 1024**3:.2f} GB")
    print(f"Swap Usage: {swap.percent}%")

def get_disk_info():
    """Get disk information"""
    print_section("DISK INFORMATION")
    
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"\nDevice: {partition.device}")
        print(f"  Mountpoint: {partition.mountpoint}")
        print(f"  File System: {partition.fstype}")
        
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            print(f"  Total Size: {usage.total / 1024**3:.2f} GB")
            print(f"  Used: {usage.used / 1024**3:.2f} GB")
            print(f"  Free: {usage.free / 1024**3:.2f} GB")
            print(f"  Usage: {usage.percent}%")
        except PermissionError:
            print("  Permission denied")

def get_network_info():
    """Get network information"""
    print_section("NETWORK INFORMATION")
    
    net_io = psutil.net_io_counters()
    print(f"Bytes Sent: {net_io.bytes_sent / 1024**3:.2f} GB")
    print(f"Bytes Received: {net_io.bytes_recv / 1024**3:.2f} GB")
    print(f"Packets Sent: {net_io.packets_sent:,}")
    print(f"Packets Received: {net_io.packets_recv:,}")

def get_gpu_processes():
    """Get GPU processes"""
    print_section("GPU PROCESSES")
    
    output = run_command("nvidia-smi pmon -c 1")
    print(output)
    
    print("\n" + "-"*80 + "\n")
    
    output = run_command("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
    print("Running GPU Processes:")
    print(output)

def get_driver_info():
    """Get driver and kernel info"""
    print_section("DRIVER & KERNEL INFORMATION")
    
    driver_version = run_command("cat /proc/driver/nvidia/version 2>/dev/null")
    if driver_version:
        print("NVIDIA Driver Info:")
        print(driver_version)
    
    print(f"\nKernel Version: {platform.release()}")
    
    # GCC version
    gcc_version = run_command("gcc --version 2>/dev/null | head -n1")
    if gcc_version:
        print(f"GCC Version: {gcc_version}")

def get_gpu_topology():
    """Get GPU topology information"""
    print_section("GPU TOPOLOGY")
    
    output = run_command("nvidia-smi topo -m")
    print(output)

def get_environment_info():
    """Get environment variables"""
    print_section("RELEVANT ENVIRONMENT VARIABLES")
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_HOME',
        'LD_LIBRARY_PATH',
        'PATH',
        'PYTHONPATH'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if var == 'PATH' or var == 'LD_LIBRARY_PATH':
            if value != 'Not set':
                print(f"\n{var}:")
                for path in value.split(':'):
                    print(f"  {path}")
        else:
            print(f"{var}: {value}")

def main():
    """Main function to run all diagnostics"""
    
    print("\n")
    print("="*80)
    print(" " * 20 + "GPU & SYSTEM DIAGNOSTICS REPORT")
    print(" " * 25 + f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check if GPU is available
    gpu_check = run_command("nvidia-smi 2>&1")
    if "NVIDIA-SMI" not in gpu_check:
        print("\n⚠️  WARNING: No NVIDIA GPU detected!")
        print("This script requires an NVIDIA GPU with drivers installed.")
        return
    
    # Run all diagnostic functions
    get_gpu_info_nvidia_smi()
    get_gpu_info_pynvml()
    get_cuda_info()
    get_driver_info()
    get_system_info()
    get_memory_info()
    get_disk_info()
    get_network_info()
    get_gpu_processes()
    get_gpu_topology()
    get_environment_info()
    
    print_section("DIAGNOSTIC COMPLETE")
    print(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
