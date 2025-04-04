import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

class MCPHandler:
    """
    Model Content Protocol Handler for directing LLM outputs to files.

    MCP Syntax:
    - User commands: "@{FILENAME.ext}" in user input will save response to file
    - Model commands: Inside a response, use:
      >>> FILE: filename.ext
      content to save
      <<<
    """
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # The regex pattern for detecting MCP in model output
        self.mcp_pattern = r'>>>\s*FILE:\s*([^\n]+)\s*\n([\s\S]*?)\n\s*<<<'

        # The pattern for user commands
        self.user_command_pattern = r'@{([^}]+)}'

        # Buffer for accumulating MCP content during streaming
        self.current_mcp_blocks = {}

    def extract_mcp_from_user_input(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract MCP commands from user input and return cleaned input.

        Args:
            user_input: The raw user input

        Returns:
            Tuple of (cleaned_input, command_dict)
        """
        commands = {}
        clean_input = user_input

        # Find all MCP commands in user input
        matches = re.findall(self.user_command_pattern, user_input)

        for match in matches:
            # Create command to save response to this file
            commands[match] = {"action": "save_response", "timestamp": datetime.now().isoformat()}

            # Remove command from input
            clean_input = clean_input.replace(f"@{{{match}}}", "")

        # Clean up any extra spaces
        clean_input = clean_input.strip()

        return clean_input, commands

    def extract_mcp_blocks(self, content: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract MCP blocks from model response.

        Args:
            content: The model's response

        Returns:
            Tuple of (cleaned_content, mcp_blocks)
        """
        # Find all MCP blocks
        matches = re.findall(self.mcp_pattern, content)
        mcp_blocks = {}

        # If no matches, return original content
        if not matches:
            return content, {}

        # Process each match
        clean_content = content
        for filename, block_content in matches:
            filename = filename.strip()
            mcp_blocks[filename] = block_content

            # Remove the block from the content
            block_text = f">>> FILE: {filename}\n{block_content}\n<<<"
            clean_content = clean_content.replace(block_text, "")

        # Clean up any extra newlines
        clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)
        clean_content = clean_content.strip()

        return clean_content, mcp_blocks

    def process_streaming_token(self, token: str, mcp_buffer: str) -> Tuple[str, str]:
        """
        Process tokens during streaming generation for MCP.

        Args:
            token: The current token from the model
            mcp_buffer: Current buffer for accumulating MCP content

        Returns:
            Tuple of (display_token, updated_buffer)
        """
        # Add token to buffer
        updated_buffer = mcp_buffer + token

        # Check if we're in or entering MCP mode
        if ">>> FILE:" in updated_buffer:
            # We're starting or in an MCP block
            if "<<<" in updated_buffer:
                # The block is complete, extract it
                try:
                    # Try to extract the complete block
                    matches = re.findall(self.mcp_pattern, updated_buffer)
                    if matches:
                        filename, content = matches[0]
                        self.current_mcp_blocks[filename.strip()] = content

                        # Reset buffer after extraction
                        return "", ""
                except:
                    # If there's an issue, just continue accumulating
                    pass

            # Still in MCP block, keep accumulating
            return "", updated_buffer

        # Not in MCP block, return token for display
        return token, updated_buffer

    def finalize_streaming(self, content: str) -> str:
        """
        Finalize streaming by processing accumulated MCP blocks.

        Args:
            content: The complete model response

        Returns:
            Cleaned response with MCP blocks removed
        """
        # Extract any MCP blocks we might have missed
        clean_content, mcp_blocks = self.extract_mcp_blocks(content)

        # Add any blocks we accumulated during streaming
        mcp_blocks.update(self.current_mcp_blocks)

        # Save all blocks to files
        if mcp_blocks:
            self.save_mcp_blocks(mcp_blocks)

        # Reset current blocks
        self.current_mcp_blocks = {}

        return clean_content

    def save_mcp_blocks(self, blocks: Dict[str, str]) -> Dict[str, bool]:
        """
        Save MCP blocks to files.

        Args:
            blocks: Dictionary mapping filenames to content

        Returns:
            Status dictionary of {filename: success_boolean}
        """
        results = {}

        for filename, content in blocks.items():
            # Ensure the filename is safe
            safe_filename = self._sanitize_filename(filename)
            file_path = os.path.join(self.output_dir, safe_filename)

            # Create subdirectories if needed
            dirname = os.path.dirname(file_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            # Save the file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                results[safe_filename] = True
                print(f"[Content saved to: {file_path}]")
            except Exception as e:
                results[safe_filename] = False
                print(f"[Error saving to {file_path}: {str(e)}]")

        return results

    def execute_commands(self, commands: Dict[str, Any]) -> Dict[str, bool]:
        """
        Execute user MCP commands.

        Args:
            commands: Command dictionary from extract_mcp_from_user_input

        Returns:
            Status dictionary
        """
        results = {}

        # Currently only supports saving to files
        for filename, cmd in commands.items():
            if cmd.get("action") == "save_response":
                # We'll handle this after response generation
                results[filename] = True

        return results

    def save_response_to_file(self, response: str, filename: str) -> bool:
        """
        Save complete response to a file (used for user commands).

        Args:
            response: The model's response
            filename: Filename to save to

        Returns:
            Success boolean
        """
        try:
            # Create safe filename
            safe_filename = self._sanitize_filename(filename)
            file_path = os.path.join(self.output_dir, safe_filename)

            # Create subdirectories if needed
            dirname = os.path.dirname(file_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            # Write response to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response)

            return True
        except Exception as e:
            print(f"[Error saving response to {filename}: {str(e)}]")
            return False

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal.

        Args:
            filename: Raw filename from MCP

        Returns:
            Safe filename
        """
        # Replace dangerous characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)

        # Prevent directory traversal attempts
        safe_name = os.path.normpath(safe_name)
        if safe_name.startswith(('/', './', '../')):
            safe_name = safe_name.lstrip('./\\')

        return safe_name

    def get_help_text(self) -> str:
        """
        Get help text explaining MCP syntax.

        Returns:
            Help text string
        """
        return """
MODEL CONTENT PROTOCOL (MCP) COMMANDS:

1. Save complete response to file:
   Type @{filename.ext} anywhere in your message.
   Example: "Explain quantum computing @{quantum.md}"

2. Save specific content to files - the model can use:
   >>> FILE: filename.ext
   Content to save goes here...
   <<<

3. Available file extensions (auto-formatted):
   - .py, .js, .html, .css: Code files
   - .md: Markdown files
   - .txt: Plain text
   - .json: JSON data
   - .csv: Comma separated values

Output files are saved to: {output_dir}
""".format(output_dir=os.path.abspath(self.output_dir))