import signal
import threading

class KeyboardInterruptHandler:
    def __init__(self):
        self.interrupt_requested = False
        self.original_handler = None
        self._lock = threading.Lock()
        
    def install(self):
        """Install the keyboard interrupt handler"""
        self.original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        self.interrupt_requested = False
        
    def uninstall(self):
        """Restore original interrupt handler"""
        if self.original_handler:
            signal.signal(signal.SIGINT, self.original_handler)
        
    def _handler(self, signum, frame):
        """Handle SIGINT (Ctrl+C)"""
        with self._lock:
            self.interrupt_requested = True
            print("\n[Generation interrupted by user (Ctrl+C)]")
    
    def check(self):
        """Check if interrupt has been requested"""
        with self._lock:
            return self.interrupt_requested
            
    def reset(self):
        """Reset interrupt flag"""
        with self._lock:
            self.interrupt_requested = False