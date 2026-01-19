"""
Advanced GPU Monitoring Dashboard with Real-Time Visualizations
For Google Colab NVIDIA T4
"""

import time
import subprocess
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
from datetime import datetime
import threading
import queue

class GPUMonitor:
    """Real-time GPU monitoring with visualization"""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.history = {
            'timestamp': [],
            'gpu_util': [],
            'mem_used': [],
            'mem_total': [],
            'temperature': [],
            'power': [],
            'sm_clock': [],
            'mem_clock': []
        }
        self.running = False
        
    def get_gpu_metrics(self):
        """Fetch current GPU metrics"""
        try:
            # Query multiple metrics at once
            query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem"
            cmd = f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'gpu_util': float(values[0]),
                    'mem_used': float(values[1]),
                    'mem_total': float(values[2]),
                    'temperature': float(values[3]),
                    'power': float(values[4]) if values[4] != '[N/A]' else 0,
                    'sm_clock': float(values[5]),
                    'mem_clock': float(values[6])
                }
        except Exception as e:
            print(f"Error fetching metrics: {e}")
        
        return None
    
    def update_history(self, metrics):
        """Update historical data"""
        if metrics:
            self.history['timestamp'].append(time.time())
            self.history['gpu_util'].append(metrics['gpu_util'])
            self.history['mem_used'].append(metrics['mem_used'])
            self.history['mem_total'].append(metrics['mem_total'])
            self.history['temperature'].append(metrics['temperature'])
            self.history['power'].append(metrics['power'])
            self.history['sm_clock'].append(metrics['sm_clock'])
            self.history['mem_clock'].append(metrics['mem_clock'])
            
            # Keep only recent history
            for key in self.history:
                if len(self.history[key]) > self.max_history:
                    self.history[key] = self.history[key][-self.max_history:]
    
    def plot_dashboard(self):
        """Create comprehensive dashboard"""
        if not self.history['timestamp']:
            return
        
        # Calculate time offsets
        start_time = self.history['timestamp'][0]
        times = [(t - start_time) for t in self.history['timestamp']]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. GPU Utilization
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(times, self.history['gpu_util'], 'b-', linewidth=2)
        ax1.fill_between(times, self.history['gpu_util'], alpha=0.3)
        ax1.set_ylabel('GPU Util (%)', fontsize=10)
        ax1.set_title('GPU Utilization', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Current value annotation
        if self.history['gpu_util']:
            current = self.history['gpu_util'][-1]
            ax1.text(0.02, 0.98, f'Current: {current:.1f}%', 
                    transform=ax1.transAxes, fontsize=10, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Memory Usage
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(times, self.history['mem_used'], 'r-', linewidth=2, label='Used')
        ax2.axhline(y=self.history['mem_total'][-1] if self.history['mem_total'] else 0, 
                   color='g', linestyle='--', linewidth=1, label='Total')
        ax2.fill_between(times, self.history['mem_used'], alpha=0.3)
        ax2.set_ylabel('Memory (MB)', fontsize=10)
        ax2.set_title('GPU Memory', fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Memory percentage
        if self.history['mem_used'] and self.history['mem_total']:
            current_used = self.history['mem_used'][-1]
            total = self.history['mem_total'][-1]
            percentage = (current_used / total * 100) if total > 0 else 0
            ax2.text(0.02, 0.98, f'Used: {current_used:.0f}/{total:.0f} MB ({percentage:.1f}%)', 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Temperature
        ax3 = plt.subplot(3, 3, 3)
        colors = ['g' if t < 65 else 'orange' if t < 80 else 'r' for t in self.history['temperature']]
        ax3.plot(times, self.history['temperature'], 'orange', linewidth=2)
        ax3.fill_between(times, self.history['temperature'], alpha=0.3, color='orange')
        ax3.set_ylabel('Temperature (°C)', fontsize=10)
        ax3.set_title('GPU Temperature', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Temperature zones
        ax3.axhspan(0, 65, facecolor='green', alpha=0.1, label='Safe')
        ax3.axhspan(65, 80, facecolor='yellow', alpha=0.1, label='Warm')
        ax3.axhspan(80, 100, facecolor='red', alpha=0.1, label='Hot')
        
        if self.history['temperature']:
            current = self.history['temperature'][-1]
            ax3.text(0.02, 0.98, f'Current: {current:.1f}°C', 
                    transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Power Usage
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(times, self.history['power'], 'purple', linewidth=2)
        ax4.fill_between(times, self.history['power'], alpha=0.3, color='purple')
        ax4.set_ylabel('Power (W)', fontsize=10)
        ax4.set_title('Power Draw', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        if self.history['power']:
            current = self.history['power'][-1]
            avg = np.mean(self.history['power']) if self.history['power'] else 0
            ax4.text(0.02, 0.98, f'Current: {current:.1f}W\nAvg: {avg:.1f}W', 
                    transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. SM Clock
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(times, self.history['sm_clock'], 'cyan', linewidth=2)
        ax5.fill_between(times, self.history['sm_clock'], alpha=0.3, color='cyan')
        ax5.set_ylabel('Clock (MHz)', fontsize=10)
        ax5.set_title('SM Clock Speed', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        if self.history['sm_clock']:
            current = self.history['sm_clock'][-1]
            ax5.text(0.02, 0.98, f'Current: {current:.0f} MHz', 
                    transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. Memory Clock
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(times, self.history['mem_clock'], 'magenta', linewidth=2)
        ax6.fill_between(times, self.history['mem_clock'], alpha=0.3, color='magenta')
        ax6.set_ylabel('Clock (MHz)', fontsize=10)
        ax6.set_title('Memory Clock Speed', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        if self.history['mem_clock']:
            current = self.history['mem_clock'][-1]
            ax6.text(0.02, 0.98, f'Current: {current:.0f} MHz', 
                    transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 7. Memory Utilization Percentage
        ax7 = plt.subplot(3, 3, 7)
        if self.history['mem_used'] and self.history['mem_total']:
            mem_percent = [(u/t*100) if t > 0 else 0 
                          for u, t in zip(self.history['mem_used'], self.history['mem_total'])]
            ax7.plot(times, mem_percent, 'brown', linewidth=2)
            ax7.fill_between(times, mem_percent, alpha=0.3, color='brown')
            ax7.set_ylabel('Utilization (%)', fontsize=10)
            ax7.set_title('Memory Utilization %', fontweight='bold')
            ax7.grid(True, alpha=0.3)
            ax7.set_ylim(0, 100)
            
            if mem_percent:
                current = mem_percent[-1]
                ax7.text(0.02, 0.98, f'Current: {current:.1f}%', 
                        transform=ax7.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 8. Statistics Summary
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        stats_text = "STATISTICS SUMMARY\n" + "="*30 + "\n\n"
        
        if self.history['gpu_util']:
            stats_text += f"GPU Utilization:\n"
            stats_text += f"  Current: {self.history['gpu_util'][-1]:.1f}%\n"
            stats_text += f"  Average: {np.mean(self.history['gpu_util']):.1f}%\n"
            stats_text += f"  Max: {np.max(self.history['gpu_util']):.1f}%\n"
            stats_text += f"  Min: {np.min(self.history['gpu_util']):.1f}%\n\n"
        
        if self.history['temperature']:
            stats_text += f"Temperature:\n"
            stats_text += f"  Current: {self.history['temperature'][-1]:.1f}°C\n"
            stats_text += f"  Average: {np.mean(self.history['temperature']):.1f}°C\n"
            stats_text += f"  Max: {np.max(self.history['temperature']):.1f}°C\n\n"
        
        if self.history['power'] and any(p > 0 for p in self.history['power']):
            valid_power = [p for p in self.history['power'] if p > 0]
            if valid_power:
                stats_text += f"Power Usage:\n"
                stats_text += f"  Current: {self.history['power'][-1]:.1f}W\n"
                stats_text += f"  Average: {np.mean(valid_power):.1f}W\n"
                stats_text += f"  Max: {np.max(valid_power):.1f}W\n"
        
        ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 9. GPU Info
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Get GPU name and info
        try:
            cmd = "nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                info = result.stdout.strip().split(', ')
                gpu_name = info[0] if len(info) > 0 else 'Unknown'
                driver = info[1] if len(info) > 1 else 'Unknown'
                compute_cap = info[2] if len(info) > 2 else 'Unknown'
                
                info_text = "GPU INFORMATION\n" + "="*30 + "\n\n"
                info_text += f"Name: {gpu_name}\n\n"
                info_text += f"Driver: {driver}\n\n"
                info_text += f"Compute Cap: {compute_cap}\n\n"
                
                if self.history['mem_total']:
                    info_text += f"Total Memory:\n{self.history['mem_total'][-1]:.0f} MB\n\n"
                
                info_text += f"Monitoring Duration:\n{times[-1]:.1f} seconds" if times else ""
                
                ax9.text(0.1, 0.9, info_text, transform=ax9.transAxes,
                        fontsize=10, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        except:
            pass
        
        # Set common x-label for bottom plots
        for ax in [ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.set_xlabel('Time (seconds)', fontsize=10)
        
        plt.suptitle(f'NVIDIA GPU Real-Time Monitoring Dashboard - {datetime.now().strftime("%H:%M:%S")}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.96)
        
        return fig
    
    def monitor(self, duration=60, interval=1):
        """Monitor GPU for specified duration"""
        print(f"Starting GPU monitoring for {duration} seconds...")
        print(f"Update interval: {interval} second(s)")
        print("-" * 80)
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Get current metrics
                metrics = self.get_gpu_metrics()
                
                if metrics:
                    self.update_history(metrics)
                    
                    # Clear output and redraw
                    display.clear_output(wait=True)
                    
                    # Plot dashboard
                    fig = self.plot_dashboard()
                    plt.show()
                    
                    # Print current status
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    print(f"\nMonitoring: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining")
                    print(f"GPU Util: {metrics['gpu_util']:.1f}% | "
                          f"Memory: {metrics['mem_used']:.0f}/{metrics['mem_total']:.0f} MB | "
                          f"Temp: {metrics['temperature']:.1f}°C | "
                          f"Power: {metrics['power']:.1f}W")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        print(f"\nMonitoring complete! Total duration: {time.time() - start_time:.1f} seconds")

# Usage example
if __name__ == "__main__":
    monitor = GPUMonitor(max_history=120)
    monitor.monitor(duration=120, interval=1)  # Monitor for 2 minutes
