import os
import readline
from datetime import datetime

def get_time():
    return datetime.now().strftime("[%d/%m/%y %H:%M:%S]")

def setup_readline_history(memory_dir):
    """Set up readline with history file support"""
    history_file = os.path.join(memory_dir, '.phistory')
    
    # Create history file if it doesn't exist
    try:
        if not os.path.exists(history_file):
            with open(history_file, 'w') as f:
                pass
            print(f"{get_time()} Created new history file: {history_file}")
    except Exception as e:
        print(f"{get_time()} Warning: Could not create history file: {e}")
        history_file = None
    
    # Set history file
    readline.read_history_file(history_file)
    
    # Set maximum number of items in history
    readline.set_history_length(1000)
    
    # Register saving history on exit
    import atexit
    atexit.register(readline.write_history_file, history_file)
    
    # Enable tab completion similar to bash
    readline.parse_and_bind('tab: complete')
    
    return history_file