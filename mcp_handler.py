import os
import re
import json
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from question_classifier import QuestionClassifier
from memory_utils import (
    classify_content,
    generate_memory_metadata
)

class MCPHandler:
    """
    Model Content Protocol Handler for directing LLM outputs to files
    and executing commands.

    MCP Syntax:
    - User commands:
      - "@{FILENAME.ext}" in user input will save response to file
      - "!{command}" in user input will execute a shell command
    - Model commands: Inside a response, use:
      >>> FILE: filename.ext
      content to save
      <<<
    """
    def __init__(self, output_dir: str = "./output", allow_shell_commands: bool = False, memory_manager = None):
        self.output_dir = output_dir
        self.allow_shell_commands = allow_shell_commands
        self.memory_manager = memory_manager
        os.makedirs(output_dir, exist_ok=True)

        # The regex pattern for detecting MCP in model output
        self.mcp_pattern = r'>>>\s*FILE:\s*([^\n]+)\s*\n([\s\S]*?)\n\s*<<<'

        # The pattern for user commands
        self.user_command_pattern = r'@{([^}]+)}'

        # The pattern for shell commands
        self.shell_command_pattern = r'!{([^}]+)}'

        # Buffer for accumulating MCP content during streaming
        self.current_mcp_blocks = {}
        self.question_classifier = QuestionClassifier()
        self.memory_manager = memory_manager

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
                print(f"[SUCCESS] Content saved to: {file_path}")
            except Exception as e:
                results[safe_filename] = False
                print(f"[ERROR] Failed saving to {file_path}: {str(e)}")

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
            print(f"[ERROR] Failed saving response to {filename}: {str(e)}")
            return False

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
        # (Shell commands are executed immediately in extract_mcp_from_user_input)
        for filename, cmd in commands.items():
            if cmd.get("action") == "save_response":
                # We'll handle this after response generation
                results[filename] = True

        return results

    def execute_response_commands(self, response: str, save_to_memory=True) -> str:
        """
        Execute MCP commands found in AI responses and replace them with outputs.

        Args:
            response: The AI's response containing MCP commands
            save_to_memory: Whether to save command outputs to memory

        Returns:
            Response with commands replaced by their outputs
        """
        if not self.allow_shell_commands:
            return response

        # Find all !{command} patterns
        import re
        command_pattern = r'!{([^}]+)}'

        def execute_and_replace(match):
            command = match.group(1)
            try:
                # Execute command
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
                output = result.stdout.strip()
                error = result.stderr.strip()

                # Save to memory if requested
                if save_to_memory and self.memory_manager and output:
                    try:
                        classification = classify_content(output, self.question_classifier)
                        metadata = generate_memory_metadata(output, classification)
                        metadata["command"] = command
                        metadata["content_type"] = self._detect_content_type(output)

                        self.memory_manager.add(
                            content=f"Command '{command}' output: {output}",
                            metadata=metadata
                        )
                    except Exception as e:
                        print(f"Error saving command output to memory: {e}")

                # Return formatted output
                if output:
                    return f"!{{{command}}} → {output}"
                elif error:
                    return f"!{{{command}}} → Error: {error}"
                else:
                    return f"!{{{command}}} → [No output]"

            except subprocess.TimeoutExpired:
                return f"!{{{command}}} → [Timed out]"
            except Exception as e:
                return f"!{{{command}}} → [Error: {str(e)}]"

        # Replace all command patterns with their outputs
        processed_response = re.sub(command_pattern, execute_and_replace, response)
        return processed_response

    def process_response_commands(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process MCP commands embedded in assistant responses.
        Similar to extract_mcp_from_user_input but handles response context.

        Args:
            response: The assistant's response text

        Returns:
            Tuple of (processed_response, command_results)
        """
        command_results = {}
        processed_response = response

        # Find all command patterns in the response
        command_matches = re.findall(self.shell_command_pattern, response)

        for match in command_matches:
            if self.allow_shell_commands:
                try:
                    # Execute the command
                    result = subprocess.run(match, shell=True, capture_output=True, text=True)
                    output = result.stdout
                    error = result.stderr

                    # Store the result
                    command_results[match] = {
                        "output": output,
                        "error": error,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Replace the command pattern with the output
                    processed_response = processed_response.replace(
                        f"!{{{match}}}",
                        f"!{{{match}}} returned:\n{output}"
                    )

                except Exception as e:
                    print(f"[Error executing command in response: {str(e)}]")

        return processed_response, command_results

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
        help_text = f"""
MODEL CONTENT PROTOCOL (MCP) COMMANDS:

1. Save complete response to file:
   Type @{{filename.ext}} anywhere in your message.
   Example: "Explain quantum computing @{{quantum.md}}"

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
"""

        # Add shell command information if enabled
        if self.allow_shell_commands:
            help_text += """
4. Execute shell commands:
   Type !{command} in your message to execute a shell command.
   Example: "List files in the current directory !{ls -la}"

   The command output will be shown immediately and included in your query.

   CAUTION: Shell commands run with your current user permissions!
"""
        else:
            help_text += """
4. Shell command execution is currently disabled.
   To enable it, initialize MCPHandler with allow_shell_commands=True
"""

        help_text += f"""
Output files are saved to: {os.path.abspath(self.output_dir)}
"""
        return help_text

    def extract_mcp_from_user_input(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract MCP commands from user input and return cleaned input.
        Also adds command outputs to memory for future reference.

        Args:
            user_input: The raw user input

        Returns:
            Tuple of (cleaned_input, command_dict)
        """
        commands = {}
        clean_input = user_input

        # Find file save commands (@{filename.ext})
        file_matches = re.findall(self.user_command_pattern, user_input)
        for match in file_matches:
            # Create command to save response to this file
            commands[match] = {"action": "save_response", "timestamp": datetime.now().isoformat()}

            # Remove command from input
            clean_input = clean_input.replace(f"@{{{match}}}", "")

        # Check if this is a shell command only request
        shell_matches = re.findall(self.shell_command_pattern, user_input)
        is_command_only = len(shell_matches) > 0 and clean_input.replace(f"!{{{shell_matches[0]}}}", "").strip() == ""

        # Process shell commands
        for match in shell_matches:
            # Execute shell command immediately if enabled
            if self.allow_shell_commands:
                try:
                    # print(f"[Executing command: {match}]")
                    result = subprocess.run(match, shell=True, capture_output=True, text=True)
                    output = result.stdout
                    error = result.stderr

                    # Save to temp file for large outputs
                    output_file = None
                    if len(output) > 1000:  # If output is large
                        output_file = os.path.join(self.output_dir, f"cmd_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                        try:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(output)
                            print(f"[Large output saved to: {output_file}]")
                        except Exception as e:
                            print(f"[Error saving output: {str(e)}]")
                            output_file = None

                    # Save to memory if memory_manager is available
                    if hasattr(self, 'memory_manager') and self.memory_manager:
                        # Detect content type
                        content_type = self._detect_content_type(output)
                        classification = classify_content(match, self.question_classifier)
                        metadata = generate_memory_metadata(match, classification)

                        # Add few more metadata for mcp types
                        metadata["command"] = match
                        metadata["output_file"] = output_file
                        metadata["content_type"] = content_type
                        metadata["error"] = error if error else None

                        # Add to memory
                        memory_id = self.memory_manager.add(
                            content=output,
                            metadata=metadata
                        )

                        if memory_id:
                            print(f"[Command output saved to memory with ID: {memory_id}]")

                        # Store memory ID for reference
                        # commands[match]["memory_id"] = memory_id

                    # Display the output (truncated for very large outputs)
                    # print("\n--- Command Output ---")
                    # if len(output) > 2000:
                    #     print(output[:1000] + "\n...[output truncated]...\n" + output[-1000:])
                    # elif output:
                    #     print(output)
                    # if error:
                    #     print("Error:", error)
                    # print("--- End Output ---\n")
                    print(output)

                    # Remove the command from the input
                    clean_input = clean_input.replace(f"!{{{match}}}", "")

                    # If this was just a command with no other text, don't send to model
                    if is_command_only:
                        # Replace with a placeholder to avoid empty input
                        clean_input = "_COMMAND_ONLY_"
                    else:
                        # Only include a summarized/truncated version of the output
                        if len(output) > 1000:

                            if output_file:
                                truncated_output = f"[Large output (saved to {output_file})]\n" + output[:500] + "\n...[truncated]..."
                            else:
                                truncated_output = output[:500] + "\n...[output truncated, total length: " + str(len(output)) + " characters]..."
                            clean_input = f"{clean_input}\n\nCommand output:\n{truncated_output}"
                        else:
                            clean_input = f"{clean_input}\n\nCommand output:\n{output}"

                except Exception as e:
                    print(f"[Error executing command: {str(e)}]")
                    import traceback
                    traceback.print_exc()
                    # Remove the command from input but leave an error message
                    clean_input = clean_input.replace(f"!{{{match}}}", f"[Command error: {str(e)}]")
            else:
                print("[Shell command execution is disabled]")
                # Remove the command from input and leave a message
                clean_input = clean_input.replace(f"!{{{match}}}", "[Shell command execution is disabled]")

        # Clean up any extra spaces and normalize newlines
        clean_input = clean_input.strip()

        return clean_input, commands

    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content from command output."""
        # Check for table-like structures (tab or multiple spaces between items)
        if '\t' in content or re.search(r'\S\s{2,}\S', content):
            return "tabular"

        # Check for filesystem info
        if any(fs in content for fs in ['Filesystem', 'Mounted on', '/dev/']):
            return "filesystem"

        # Check for JSON structure
        if content.strip().startswith('{') and content.strip().endswith('}'):
            return "json"

        # Check for code (simple heuristic)
        code_indicators = ['def ', 'function', 'class ', 'import ', 'from ', '#!/bin/']
        if any(indicator in content for indicator in code_indicators):
            return "code"

        # Default
        return "text"