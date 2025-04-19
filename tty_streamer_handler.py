import sys
import os
import select
import tty
import termios
import time
import re

class TTYStreamHandler:
    """
    Handles streaming output to TTY with proper token joining and alignment.
    """

    def __init__(self, colorizer=None, buffer_size=16, flush_interval=0.05, stop_sequences=None):
        self.colorizer = colorizer
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.stop_sequences = stop_sequences or []

        self.buffer = ""
        self.last_flush_time = time.time()
        self.complete_output = ""
        self.stopped = False

        # Position tracking
        self.line_position = 0

        # Terminal state
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

    def process_token(self, token, confidence=None):
        """Process a token for display with proper alignment."""
        if self.stopped:
            return False

        # Add to complete output
        self.complete_output += token

        # Check for stop sequences
        for stop_seq in self.stop_sequences:
            if stop_seq in self.complete_output:
                self.complete_output = self.complete_output.split(stop_seq)[0]
                self.stopped = True
                self._flush_buffer()
                return False

        # Apply colorization if needed
        display_token = token
        if self.colorizer and confidence is not None:
            display_token = self.colorizer(token, confidence)

        # The key fix: normalize whitespace at token boundaries
        # This ensures tokens join properly without strange indentation
        if self.buffer and self.buffer[-1].isspace() and display_token and display_token[0].isspace():
            # If both buffer ends with space and new token starts with space,
            # normalize to just one space to avoid double spacing
            display_token = display_token.lstrip()

        # Handle special characters
        i = 0
        while i < len(display_token):
            char = display_token[i]

            if char == '\n':
                # Flush buffer and reset position on newline
                self._flush_buffer()
                sys.stdout.write('\n')
                sys.stdout.flush()
                self.line_position = 0
            elif char == '\r':
                # Flush buffer and reset position on carriage return
                self._flush_buffer()
                sys.stdout.write('\r')
                sys.stdout.flush()
                self.line_position = 0
            elif char == '\t':
                # Calculate tab stop position (usually every 8 characters)
                tab_size = 8 - (self.line_position % 8)
                self._flush_buffer()
                sys.stdout.write(' ' * tab_size)
                sys.stdout.flush()
                self.line_position += tab_size
            else:
                # Regular character - add to buffer
                self.buffer += char
                self.line_position += 1

                # Auto-flush on buffer size
                if len(self.buffer) >= self.buffer_size:
                    self._flush_buffer()

            i += 1

        # Time-based flush
        current_time = time.time()
        if current_time - self.last_flush_time >= self.flush_interval:
            self._flush_buffer()

        # Check for interrupt
        if select.select([self.fd], [], [], 0)[0]:
            c = sys.stdin.read(1)
            if c == '\x03':  # Ctrl+C
                self.stopped = True
                return False

        return True

    def _flush_buffer(self):
        """Flush buffer to stdout and update position."""
        if self.buffer:
            sys.stdout.write(self.buffer)
            sys.stdout.flush()
            self.buffer = ""
            self.last_flush_time = time.time()

    def finalize(self):
        """Finalize and return complete output."""
        self._flush_buffer()
        return self.complete_output

    def check_interrupt(self):
        """Check for user interrupt."""
        if select.select([self.fd], [], [], 0)[0]:
            c = sys.stdin.read(1)
            if c == '\x03':  # Ctrl+C
                self.stopped = True
                return True
        return False