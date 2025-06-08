"""
Real-time performance monitor for debugging generation issues
Run this alongside your chat to see what's happening
"""

import torch
import time
import threading
import psutil
import os
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.running = False
        self.stats = {
            'tokens_generated': 0,
            'generation_time': 0,
            'last_token_time': None,
            'token_times': []
        }
    
    def start(self):
        """Start monitoring in background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process(os.getpid())
        
        while self.running:
            try:
                # CPU and memory
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_info = process.memory_info()
                
                # GPU stats if available
                gpu_stats = ""
                if torch.cuda.is_available():
                    try:
                        gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
                        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        # Comment out gpu_util as it might trigger the AMD issue
                        # gpu_util = torch.cuda.utilization()
                        gpu_stats = f" | GPU: {gpu_mem_used:.1f}/{gpu_mem_total:.1f}GB"
                    except Exception as e:
                        gpu_stats = " | GPU: Error"
                
                # Token generation rate
                if self.stats['tokens_generated'] > 0:
                    avg_time = self.stats['generation_time'] / self.stats['tokens_generated']
                    tokens_per_sec = 1.0 / avg_time if avg_time > 0 else 0
                    token_stats = f" | {tokens_per_sec:.1f} tok/s"
                else:
                    token_stats = " | Waiting..."
                
                # Print status line
                status = f"\r[{datetime.now().strftime('%H:%M:%S')}] CPU: {cpu_percent:5.1f}% | RAM: {memory_info.rss/1024**3:.1f}GB{gpu_stats}{token_stats}"
                print(status, end='', flush=True)
                
                time.sleep(1)
                
            except Exception as e:
                print(f"\nMonitor error: {e}")
                break
    
    def on_token_generated(self):
        """Call this when a token is generated"""
        now = time.time()
        if self.stats['last_token_time'] is not None:
            token_time = now - self.stats['last_token_time']
            self.stats['token_times'].append(token_time)
            self.stats['generation_time'] += token_time
            
            # Keep only last 100 token times
            if len(self.stats['token_times']) > 100:
                self.stats['token_times'].pop(0)
        
        self.stats['last_token_time'] = now
        self.stats['tokens_generated'] += 1


# Integration with streaming generation
def add_performance_monitoring(chat_instance):
    """Add performance monitoring to chat instance"""
    
    monitor = PerformanceMonitor()
    chat_instance._perf_monitor = monitor
    
    # Patch the streaming to include monitoring
    original_generate = chat_instance._generate_with_streaming
    
    def monitored_generate(*args, **kwargs):
        monitor.start()
        try:
            # Add token callback if possible
            if 'streamer' in kwargs:
                original_on_finalized = kwargs['streamer'].on_finalized_text
                
                def monitored_on_finalized(text, **kw):
                    monitor.on_token_generated()
                    return original_on_finalized(text, **kw)
                
                kwargs['streamer'].on_finalized_text = monitored_on_finalized
            
            return original_generate(*args, **kwargs)
        finally:
            monitor.stop()
            print()  # New line after monitoring
    
    chat_instance._generate_with_streaming = monitored_generate
    
    print("[Monitor] Performance monitoring enabled")
    return chat_instance


# Standalone monitoring script
if __name__ == "__main__":
    print("=== Standalone Performance Monitor ===")
    print("This will monitor system resources while your model runs")
    print("Press Ctrl+C to stop\n")
    
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("\n\nMonitoring stopped.")