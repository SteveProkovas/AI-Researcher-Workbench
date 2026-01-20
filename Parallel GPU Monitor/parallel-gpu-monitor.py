import threading
import queue
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import csv
from pathlib import Path
import signal
import sys
import os


class ParallelGPUMonitor:
    """
    Professional GPU monitor that runs in a separate thread
    Allows monitoring while your main code executes
    THREAD-SAFE with proper synchronization
    """

    def __init__(self, update_interval=1, log_to_file=True, log_dir="gpu_logs",
                 gpu_id=0, max_queue_size=1000):
        """
        Initialize parallel GPU monitor

        Args:
            update_interval: Seconds between updates (default: 1)
            log_to_file: Whether to save metrics to CSV (default: True)
            log_dir: Directory for log files (default: "gpu_logs")
            gpu_id: GPU device ID to monitor (default: 0)
            max_queue_size: Maximum queue size to prevent memory leaks (default: 1000)
        """
        self.update_interval = update_interval
        self.log_to_file = log_to_file
        self.log_dir = Path(log_dir)
        self.gpu_id = gpu_id

        # Thread control
        self.monitoring = False
        self.monitor_thread = None
        self.data_lock = threading.Lock()  # FIX: Thread safety

        # Data storage with bounded queue (FIX: Memory leak prevention)
        self.metrics_queue = queue.Queue(maxsize=max_queue_size)
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

        # Statistics (protected by lock)
        self.stats = {
            'peak_gpu_util': 0,
            'peak_memory': 0,
            'peak_temperature': 0,
            'avg_gpu_util': 0,
            'avg_temperature': 0,
            'sum_gpu_util': 0,  # FIX: Running sum for efficient average
            'sum_temperature': 0,
            'sample_count': 0
        }

        # File handles (FIX: Keep file open during monitoring)
        self.csv_filehandle = None
        self.csv_writer = None

        # Log file paths
        if self.log_to_file:
            self.log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_file = self.log_dir / f"gpu_metrics_{timestamp}.csv"
            self.json_file = self.log_dir / f"gpu_stats_{timestamp}.json"

    def _get_gpu_metrics(self):
        """Fetch current GPU metrics"""
        try:
            query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem"
            # FIX: Specify GPU ID
            cmd = f"nvidia-smi -i {self.gpu_id} --query-gpu={query} --format=csv,noheader,nounits"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'timestamp': time.time(),
                    'gpu_util': float(values[0]),
                    'mem_used': float(values[1]),
                    'mem_total': float(values[2]),
                    'temperature': float(values[3]),
                    'power': float(values[4]) if values[4] != '[N/A]' else 0,
                    'sm_clock': float(values[5]),
                    'mem_clock': float(values[6])
                }
            else:
                print(f"⚠️  nvidia-smi error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("⚠️  nvidia-smi timeout")
        except Exception as e:
            print(f"⚠️  Error fetching GPU metrics: {e}")

        return None

    def _monitoring_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        print(f"🚀 GPU monitoring started (GPU {self.gpu_id}, interval: {self.update_interval}s)")

        # Initialize CSV file (FIX: Keep file handle open)
        if self.log_to_file:
            try:
                self.csv_filehandle = open(self.csv_file, 'w', newline='', buffering=1)  # Line buffering
                self.csv_writer = csv.writer(self.csv_filehandle)
                self.csv_writer.writerow(['timestamp', 'gpu_util', 'mem_used', 'mem_total',
                                         'temperature', 'power', 'sm_clock', 'mem_clock'])
            except IOError as e:
                print(f"⚠️  Cannot create log file: {e}")
                self.log_to_file = False

        start_time = time.time()

        while self.monitoring:
            metrics = self._get_gpu_metrics()

            if metrics:
                # Add to queue (FIX: Handle full queue)
                try:
                    self.metrics_queue.put_nowait(metrics)
                except queue.Full:
                    # Remove oldest item and add new one
                    try:
                        self.metrics_queue.get_nowait()
                        self.metrics_queue.put_nowait(metrics)
                    except:
                        pass

                # Update history and statistics (FIX: Thread-safe access)
                with self.data_lock:
                    # Update history
                    for key in self.history:
                        if key in metrics:
                            self.history[key].append(metrics[key])
                        elif key == 'timestamp':
                            self.history[key].append(metrics['timestamp'] - start_time)

                    # Update statistics (FIX: Use running sums for efficiency)
                    self.stats['peak_gpu_util'] = max(self.stats['peak_gpu_util'], metrics['gpu_util'])
                    self.stats['peak_memory'] = max(self.stats['peak_memory'], metrics['mem_used'])
                    self.stats['peak_temperature'] = max(self.stats['peak_temperature'], metrics['temperature'])

                    self.stats['sum_gpu_util'] += metrics['gpu_util']
                    self.stats['sum_temperature'] += metrics['temperature']
                    self.stats['sample_count'] += 1

                    # Calculate averages
                    self.stats['avg_gpu_util'] = self.stats['sum_gpu_util'] / self.stats['sample_count']
                    self.stats['avg_temperature'] = self.stats['sum_temperature'] / self.stats['sample_count']

                # Log to CSV (FIX: File already open, just write)
                if self.log_to_file and self.csv_writer:
                    try:
                        self.csv_writer.writerow([
                            metrics['timestamp'],
                            metrics['gpu_util'],
                            metrics['mem_used'],
                            metrics['mem_total'],
                            metrics['temperature'],
                            metrics['power'],
                            metrics['sm_clock'],
                            metrics['mem_clock']
                        ])
                    except IOError as e:
                        print(f"⚠️  Error writing to log: {e}")

            time.sleep(self.update_interval)

        # Close CSV file handle
        if self.csv_filehandle:
            self.csv_filehandle.close()

        # Save final statistics
        if self.log_to_file:
            self._save_statistics()

        print("✅ GPU monitoring stopped")

    def start(self):
        """Start monitoring in background thread"""
        if self.monitoring:
            print("⚠️  Monitoring already running")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print(f"✅ Monitoring started in background thread")

    def stop(self):
        """Stop monitoring"""
        if not self.monitoring:
            print("⚠️  Monitoring not running")
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("✅ Monitoring stopped")

    def get_current_metrics(self):
        """Get most recent metrics (non-blocking)"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None

    def get_stats(self):
        """Get current statistics (thread-safe)"""
        with self.data_lock:
            return self.stats.copy()

    def get_history_snapshot(self):
        """Get a snapshot of history data (thread-safe)"""
        with self.data_lock:
            return {key: list(values) for key, values in self.history.items()}

    def _save_statistics(self):
        """Save statistics to JSON file"""
        with self.data_lock:
            duration = self.history['timestamp'][-1] if self.history['timestamp'] else 0

            stats_summary = {
                'gpu_id': self.gpu_id,
                'monitoring_duration_seconds': duration,
                'samples_collected': len(self.history['gpu_util']),
                'peak_gpu_utilization': self.stats['peak_gpu_util'],
                'average_gpu_utilization': self.stats['avg_gpu_util'],
                'peak_memory_usage_mb': self.stats['peak_memory'],
                'peak_temperature_celsius': self.stats['peak_temperature'],
                'average_temperature_celsius': self.stats['avg_temperature'],
                'csv_log_file': str(self.csv_file),
                'update_interval_seconds': self.update_interval
            }

        try:
            with open(self.json_file, 'w') as f:
                json.dump(stats_summary, f, indent=2)
            print(f"📊 Statistics saved to: {self.json_file}")
        except IOError as e:
            print(f"⚠️  Cannot save statistics: {e}")

    def plot_live(self, figsize=(14, 8)):
        """Display live monitoring dashboard (thread-safe, works in Jupyter)"""
        # FIX: Check if running in Jupyter
        try:
            from IPython import display
            in_jupyter = True
        except:
            in_jupyter = False

        history = self.get_history_snapshot()

        if not history['timestamp']:
            print("No data collected yet...")
            return

        if in_jupyter:
            display.clear_output(wait=True)

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Live GPU Monitor (GPU {self.gpu_id}) - {datetime.now().strftime("%H:%M:%S")} - {len(history["timestamp"])} samples',
                    fontsize=14, fontweight='bold')

        times = history['timestamp']
        stats = self.get_stats()

        # GPU Utilization
        axes[0, 0].plot(times, history['gpu_util'], 'b-', linewidth=2)
        axes[0, 0].fill_between(times, history['gpu_util'], alpha=0.3)
        axes[0, 0].set_ylabel('GPU Util (%)')
        axes[0, 0].set_title('GPU Utilization')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].axhline(y=stats['avg_gpu_util'], color='r', linestyle='--',
                          alpha=0.5, label=f'Avg: {stats["avg_gpu_util"]:.1f}%')
        axes[0, 0].legend()

        # Memory Usage
        axes[0, 1].plot(times, history['mem_used'], 'r-', linewidth=2, label='Used')
        if history['mem_total']:
            axes[0, 1].axhline(y=history['mem_total'][-1], color='g',
                              linestyle='--', label='Total')
        axes[0, 1].fill_between(times, history['mem_used'], alpha=0.3)
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].set_title('GPU Memory')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Temperature
        axes[0, 2].plot(times, history['temperature'], 'orange', linewidth=2)
        axes[0, 2].fill_between(times, history['temperature'], alpha=0.3, color='orange')
        axes[0, 2].set_ylabel('Temperature (°C)')
        axes[0, 2].set_title('GPU Temperature')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=stats['avg_temperature'], color='r', linestyle='--',
                          alpha=0.5, label=f'Avg: {stats["avg_temperature"]:.1f}°C')
        axes[0, 2].legend()

        # Power
        if any(p > 0 for p in history['power']):
            axes[1, 0].plot(times, history['power'], 'purple', linewidth=2)
            axes[1, 0].fill_between(times, history['power'], alpha=0.3, color='purple')
            axes[1, 0].set_ylabel('Power (W)')
            axes[1, 0].set_title('Power Draw')
            axes[1, 0].grid(True, alpha=0.3)

        # SM Clock
        axes[1, 1].plot(times, history['sm_clock'], 'cyan', linewidth=2)
        axes[1, 1].set_ylabel('Clock (MHz)')
        axes[1, 1].set_title('SM Clock Speed')
        axes[1, 1].grid(True, alpha=0.3)

        # Statistics
        axes[1, 2].axis('off')
        stats_text = f"""
LIVE STATISTICS

Peak GPU: {stats['peak_gpu_util']:.1f}%
Avg GPU: {stats['avg_gpu_util']:.1f}%

Peak Mem: {stats['peak_memory']:.0f} MB
Current: {history['mem_used'][-1]:.0f} MB

Peak Temp: {stats['peak_temperature']:.1f}°C
Avg Temp: {stats['avg_temperature']:.1f}°C

Runtime: {times[-1]:.1f}s
Samples: {len(times)}
"""
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=11, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        plt.show()

    def print_live_terminal(self):
        """Print live statistics to terminal (works anywhere)"""
        history = self.get_history_snapshot()
        stats = self.get_stats()

        if not history['timestamp']:
            print("No data collected yet...")
            return

        # Clear terminal (cross-platform)
        os.system('clear' if os.name == 'posix' else 'cls')

        print("=" * 80)
        print(f"  GPU LIVE MONITOR - GPU {self.gpu_id} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        current_util = history['gpu_util'][-1]
        current_mem = history['mem_used'][-1]
        current_temp = history['temperature'][-1]
        total_mem = history['mem_total'][-1]

        # Current metrics
        print(f"\n📊 CURRENT METRICS:")
        print(f"   GPU Utilization: {current_util:>6.1f}% {self._bar(current_util, 100, 30)}")
        print(f"   Memory Usage:    {current_mem:>6.0f} MB / {total_mem:.0f} MB {self._bar(current_mem, total_mem, 30)}")
        print(f"   Temperature:     {current_temp:>6.1f}°C {self._temp_indicator(current_temp)}")

        if history['power'] and history['power'][-1] > 0:
            print(f"   Power Draw:      {history['power'][-1]:>6.1f} W")

        # Statistics
        print(f"\n📈 STATISTICS:")
        print(f"   Peak GPU Util:   {stats['peak_gpu_util']:>6.1f}%")
        print(f"   Avg GPU Util:    {stats['avg_gpu_util']:>6.1f}%")
        print(f"   Peak Memory:     {stats['peak_memory']:>6.0f} MB")
        print(f"   Peak Temp:       {stats['peak_temperature']:>6.1f}°C")
        print(f"   Avg Temp:        {stats['avg_temperature']:>6.1f}°C")

        # Runtime info
        runtime = history['timestamp'][-1]
        print(f"\n⏱️  RUNTIME INFO:")
        print(f"   Duration:        {runtime:>6.1f}s ({runtime/60:.1f} min)")
        print(f"   Samples:         {len(history['timestamp'])}")
        print(f"   Update Interval: {self.update_interval}s")

        if self.log_to_file:
            print(f"\n📁 LOGGING:")
            print(f"   CSV: {self.csv_file.name}")

        print("\n" + "=" * 80)
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)

    def _bar(self, value, max_value, width=20):
        """Create a simple text progress bar"""
        filled = int((value / max_value) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"

    def _temp_indicator(self, temp):
        """Temperature status indicator"""
        if temp < 60:
            return "❄️  Cool"
        elif temp < 75:
            return "🌡️  Normal"
        elif temp < 85:
            return "🔥 Warm"
        else:
            return "⚠️  Hot!"

    def generate_report(self, save_plot=True):
        """Generate final monitoring report (thread-safe)"""
        print("\n" + "="*80)
        print(f"  GPU MONITORING REPORT - GPU {self.gpu_id}")
        print("="*80)

        history = self.get_history_snapshot()
        stats = self.get_stats()

        if not history['timestamp']:
            print("No data collected")
            return

        duration = history['timestamp'][-1]

        print(f"\n📊 Monitoring Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"📈 Samples Collected: {len(history['timestamp'])}")
        print(f"⏱️  Update Interval: {self.update_interval} seconds")

        print(f"\n🎮 GPU Utilization:")
        print(f"   Average: {stats['avg_gpu_util']:.2f}%")
        print(f"   Peak: {stats['peak_gpu_util']:.2f}%")
        print(f"   Min: {min(history['gpu_util']):.2f}%")

        print(f"\n💾 Memory Usage:")
        print(f"   Peak: {stats['peak_memory']:.0f} MB")
        print(f"   Average: {np.mean(history['mem_used']):.0f} MB")
        print(f"   Total Capacity: {history['mem_total'][-1]:.0f} MB")

        print(f"\n🌡️  Temperature:")
        print(f"   Average: {stats['avg_temperature']:.2f}°C")
        print(f"   Peak: {stats['peak_temperature']:.2f}°C")
        print(f"   Min: {min(history['temperature']):.2f}°C")

        if self.log_to_file:
            print(f"\n📁 Log Files:")
            print(f"   CSV: {self.csv_file}")
            print(f"   JSON: {self.json_file}")

        if save_plot:
            plot_file = self.log_dir / f"gpu_monitoring_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.plot_live(figsize=(16, 10))
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Plot: {plot_file}")

        print("="*80 + "\n")


# ============================================================================
# STANDALONE LIVE MONITORING MODE
# ============================================================================

class StandaloneLiveMonitor:
    """
    NEW: Standalone monitor that runs continuously until stopped
    Perfect for running in a separate terminal while training
    """

    def __init__(self, update_interval=2, gpu_id=0, log_to_file=True, log_dir="gpu_logs"):
        """
        Initialize standalone live monitor

        Args:
            update_interval: Seconds between display updates
            gpu_id: GPU device ID to monitor
            log_to_file: Whether to save logs
            log_dir: Directory for logs
        """
        self.monitor = ParallelGPUMonitor(
            update_interval=update_interval,
            gpu_id=gpu_id,
            log_to_file=log_to_file,
            log_dir=log_dir
        )
        self.running = True

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Stopping monitor...")
        self.running = False

    def run(self):
        """Run continuous live monitoring"""
        print("="*80)
        print("  STANDALONE GPU LIVE MONITOR")
        print("="*80)
        print(f"\n🚀 Starting continuous monitoring of GPU {self.monitor.gpu_id}")
        print(f"⏱️  Update interval: {self.monitor.update_interval}s")
        print(f"📁 Logs: {self.monitor.log_dir}")
        print("\nPress Ctrl+C to stop\n")
        time.sleep(2)

        # Start monitoring
        self.monitor.start()

        # Continuous display loop
        try:
            while self.running:
                self.monitor.print_live_terminal()
                time.sleep(self.monitor.update_interval)
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup
            print("\n\n🛑 Stopping monitor...")
            self.monitor.stop()
            time.sleep(1)

            # Generate final report
            self.monitor.generate_report(save_plot=True)

            print("\n✅ Monitoring session complete!")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_parallel_monitoring():
    """Example: Monitor GPU while training a model"""

    print("Example: Parallel Monitoring During Model Training")
    print("="*80)

    # Create monitor
    monitor = ParallelGPUMonitor(update_interval=2, log_to_file=True, gpu_id=0)

    # Start monitoring in background
    monitor.start()

    # Simulate training loop
    print("\n🏋️  Starting training simulation...")
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/5")

        # Simulate training batches
        for batch in range(10):
            # Your actual training code would go here
            time.sleep(0.5)  # Simulating computation

            # Optionally check GPU metrics during training
            metrics = monitor.get_current_metrics()
            if metrics and batch % 5 == 0:
                print(f"  Batch {batch}: GPU {metrics['gpu_util']:.1f}%, "
                      f"Mem {metrics['mem_used']:.0f}MB, "
                      f"Temp {metrics['temperature']:.1f}°C")

        # Show live dashboard every epoch (if in Jupyter)
        print("\n📊 Current GPU status:")
        try:
            monitor.plot_live(figsize=(12, 6))
        except:
            # If not in Jupyter, print terminal stats
            monitor.print_live_terminal()
        time.sleep(1)

    print("\n✅ Training complete!")

    # Stop monitoring
    monitor.stop()

    # Generate report
    monitor.generate_report(save_plot=True)


def example_simple_background_monitoring():
    """Example: Simple background monitoring"""

    # Start monitor
    monitor = ParallelGPUMonitor(update_interval=1, gpu_id=0)
    monitor.start()

    # Your code runs here
    print("Running your code while monitoring in background...")
    time.sleep(10)

    # Check stats anytime
    stats = monitor.get_stats()
    print(f"\nCurrent stats: {stats}")

    # Stop when done
    monitor.stop()
    monitor.generate_report()


def run_standalone_monitor(update_interval=2, gpu_id=0):
    """
    NEW: Run standalone live monitor
    This runs continuously until you press Ctrl+C
    Perfect for running in a separate terminal!
    """
    monitor = StandaloneLiveMonitor(
        update_interval=update_interval,
        gpu_id=gpu_id,
        log_to_file=True
    )
    monitor.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='GPU Monitor - Professional GPU monitoring tool')
    parser.add_argument('--mode', choices=['standalone', 'example'], default='example',
                       help='Run mode: standalone (continuous) or example (demo)')
    parser.add_argument('--interval', type=float, default=2,
                       help='Update interval in seconds (default: 2)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to monitor (default: 0)')

    # Fix for Jupyter/Colab environments: parse empty args list
    args = parser.parse_args([])
    
    if args.mode == 'standalone':
        # Run continuous live monitor
        run_standalone_monitor(update_interval=args.interval, gpu_id=args.gpu)
    else:
        # Run example
        example_parallel_monitoring()
