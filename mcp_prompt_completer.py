"""
MCP Command completion for prompt_toolkit.
Provides completion for @{filename} and !{command} patterns.
"""

import os
import re
import subprocess
from typing import Iterable, List, Dict, Optional, Tuple, Union

from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document

class MCPCompleter(Completer):
    """
    Custom completer for MCP commands.
    Handles '@{filename}' for file saving and '!{command}' for shell commands.
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the completer.
        
        Args:
            output_dir: Base directory for file output
        """
        self.output_dir = output_dir
        
        # Regular expressions for matching MCP patterns
        self.file_pattern = re.compile(r'@{([^}]*)')
        self.command_pattern = re.compile(r'!{([^}]*)')
        
        # Create sub-completers
        self.path_completer = PathCompleter(
            get_paths=lambda: [self.output_dir], 
            file_filter=None,
            min_input_len=0
        )
        
        # Cache for executable commands to avoid repeated PATH searches
        self.executable_cache: Dict[str, List[str]] = {}
    
    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """
        Get completion suggestions for the current document.
        
        Args:
            document: Current document being edited
            complete_event: Completion event details
            
        Returns:
            Iterable of completions
        """
        text = document.text_before_cursor
        
        # Check for file completion (@{path...)
        file_match = self.file_pattern.search(text)
        if file_match:
            file_prefix = file_match.group(1)
            file_start_pos = len(text) - len(file_prefix)
            
            # Get file completions relative to output directory
            return self._get_file_completions(document, file_prefix, file_start_pos)
            
        # Check for command completion (!{command...)
        command_match = self.command_pattern.search(text)
        if command_match:
            command_prefix = command_match.group(1)
            command_start_pos = len(text) - len(command_prefix)
            
            # Get command completions
            return self._get_command_completions(document, command_prefix, command_start_pos)
            
        # No MCP pattern detected
        return []
    
    def _get_file_completions(self, document: Document, prefix: str, start_pos: int) -> Iterable[Completion]:
        """
        Get file path completions.
        
        Args:
            document: Current document
            prefix: Current path prefix
            start_pos: Start position for replacement
            
        Returns:
            Iterable of file completions
        """
        # Determine the base directory and relative path
        if prefix.startswith('/'):
            # Absolute path
            base_dir = '/'
            rel_path = prefix.lstrip('/')
        else:
            # Relative to output directory
            base_dir = self.output_dir
            rel_path = prefix
        
        # Build a path to search
        if '/' in rel_path:
            dir_part, file_part = os.path.split(rel_path)
            search_dir = os.path.join(base_dir, dir_part)
            prefix_to_complete = file_part
        else:
            search_dir = base_dir
            prefix_to_complete = rel_path
            
        # Ensure directory exists
        if not os.path.isdir(search_dir):
            return []
        
        # List matching files/directories
        matches = []
        try:
            for item in os.listdir(search_dir):
                if item.startswith(prefix_to_complete):
                    full_path = os.path.join(search_dir, item)
                    
                    # Get display text and replacement 
                    if os.path.isdir(full_path):
                        # For directories, add trailing slash
                        display = f"{item}/"
                        replacement = display
                    else:
                        display = item
                        replacement = item
                        
                    # For relative paths, build relative to output dir
                    if base_dir == self.output_dir:
                        if '/' in rel_path:
                            replacement = os.path.join(dir_part, replacement)
                    
                    matches.append((display, replacement))
        except (PermissionError, FileNotFoundError):
            return []
            
        # Return completions
        for display, replacement in matches:
            yield Completion(
                replacement, 
                start_position=-len(prefix),
                display=display,
                display_meta="dir" if display.endswith('/') else "file"
            )
            
        # Add option to complete with closing brace when no more matches
        if matches:
            yield Completion(
                "}", 
                start_position=0,
                display="Close", 
                display_meta="Close tag"
            )
    
    def _get_command_completions(self, document: Document, prefix: str, start_pos: int) -> Iterable[Completion]:
        """
        Get shell command completions.
        
        Args:
            document: Current document
            prefix: Current command prefix
            start_pos: Start position for replacement
            
        Returns:
            Iterable of command completions
        """
        # Split command into words
        words = prefix.split()
        
        # No words or space at end means completing a new argument
        if not words or prefix.endswith(' '):
            current_word = ""
            command_prefix = prefix
        else:
            current_word = words[-1]
            command_prefix = prefix[:-len(current_word)]
            
        # If first word, complete executable commands
        if len(words) <= 1 and not prefix.endswith(' '):
            for cmd, meta in self._get_executables(current_word):
                yield Completion(
                    cmd, 
                    start_position=-len(current_word),
                    display=cmd,
                    display_meta=meta
                )
                
        # Otherwise complete file paths for arguments
        else:
            # Determine base directory
            if current_word.startswith('/'):
                base_dir = '/'
                current_path = current_word
            else:
                base_dir = os.getcwd()
                current_path = current_word
                
            # Get file completions
            if '/' in current_path:
                dir_part, file_part = os.path.split(current_path)
                search_dir = os.path.join(base_dir, dir_part)
                search_prefix = file_part
            else:
                search_dir = base_dir
                search_prefix = current_path
                
            # List matching files/directories 
            if os.path.isdir(search_dir):
                try:
                    for item in os.listdir(search_dir):
                        if item.startswith(search_prefix):
                            full_path = os.path.join(search_dir, item)
                            
                            # Format display and replacement
                            if os.path.isdir(full_path):
                                display = f"{item}/"
                                replacement = display
                            else:
                                display = item
                                replacement = item
                                
                            # For relative paths, build full path
                            if current_path:
                                if '/' in current_path:
                                    replacement = os.path.join(dir_part, replacement)
                                
                            # Yield completion
                            yield Completion(
                                replacement, 
                                start_position=-len(current_word),
                                display=display,
                                display_meta="dir" if display.endswith('/') else "file"
                            )
                except (PermissionError, FileNotFoundError):
                    pass
                    
        # Add option to complete with closing brace
        yield Completion(
            "}", 
            start_position=0,
            display="Close", 
            display_meta="Close tag"
        )
    
    def _get_executables(self, prefix: str) -> List[Tuple[str, str]]:
        """
        Get executable commands matching the prefix.
        
        Args:
            prefix: Command prefix to match
            
        Returns:
            List of (command, metadata) tuples
        """
        # Check cache first
        cache_key = prefix[:1] if prefix else ""
        if cache_key in self.executable_cache:
            # Filter cached results by current prefix
            return [(cmd, meta) for cmd, meta in self.executable_cache[cache_key] 
                   if cmd.startswith(prefix)]
        
        # Build list of executables
        executables = []
        
        # Common shell built-ins
        builtins = {
            'cd': 'Change directory',
            'ls': 'List directory contents',
            'mkdir': 'Make directory',
            'rm': 'Remove files/directories',
            'cp': 'Copy files',
            'mv': 'Move files',
            'cat': 'Display file contents',
            'echo': 'Display text',
            'grep': 'Search text',
            'find': 'Find files',
            'awk': 'Text processing',
            'sed': 'Stream editor',
            'pwd': 'Current directory',
            'touch': 'Update file timestamp',
            'chmod': 'Change file permissions',
            'chown': 'Change file owner',
            'ps': 'Process status',
            'top': 'Monitor processes',
            'kill': 'Kill processes',
            'history': 'Command history',
            'alias': 'Define command alias',
            'export': 'Set environment variable',
            'source': 'Run script in current shell',
            'ssh': 'Secure shell',
            'scp': 'Secure copy',
            'curl': 'Transfer data',
            'wget': 'Download files',
            'tar': 'Archive utility',
            'gzip': 'Compression tool',
            'unzip': 'Extract zip files',
            'less': 'View file contents',
            'more': 'View file contents',
            'head': 'Show file beginning',
            'tail': 'Show file end',
            'du': 'Disk usage',
            'df': 'Disk free space',
            'mount': 'Mount filesystem',
            'umount': 'Unmount filesystem',
            'ping': 'Network test',
            'ifconfig': 'Network interfaces',
            'ip': 'IP configuration',
            'netstat': 'Network statistics',
            'git': 'Version control',
            'python': 'Python interpreter',
            'python3': 'Python 3 interpreter',
            'pip': 'Python package manager',
            'npm': 'Node.js package manager',
            'node': 'Node.js runtime',
            'docker': 'Container platform'
        }
        
        # Add built-ins that match prefix
        for cmd, desc in builtins.items():
            if cmd.startswith(prefix):
                executables.append((cmd, desc))
        
        # Search PATH for executables
        paths = os.environ.get('PATH', '').split(os.pathsep)
        for path in paths:
            if not os.path.isdir(path):
                continue
                
            try:
                for item in os.listdir(path):
                    if not item.startswith(prefix):
                        continue
                        
                    full_path = os.path.join(path, item)
                    if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                        # Check if this is already in our list
                        if not any(cmd == item for cmd, _ in executables):
                            executables.append((item, f"Executable in {path}"))
            except (PermissionError, FileNotFoundError):
                continue
        
        # Sort by command name
        executables.sort(key=lambda x: x[0])
        
        # Cache results
        self.executable_cache[cache_key] = executables
        
        return executables

def create_mcp_completer(output_dir: str = "./output") -> MCPCompleter:
    """
    Create an MCP completer instance.
    
    Args:
        output_dir: Directory for output files
        
    Returns:
        Configured MCPCompleter instance
    """
    return MCPCompleter(output_dir)