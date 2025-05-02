import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_handler import MCPHandler

def test_mcp():
    """Test Model Content Protocol (MCP) functionality"""

    # Initialize MCP handler
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)
    mcp = MCPHandler(output_dir=output_dir, allow_shell_commands=True)

    print("Testing MCP Handler functionality...")

    # Test user command extraction
    print("\n1. Testing user command extraction")

    user_inputs = [
        "Can you explain how Python works @{python_explanation.md}",
        "Write a simple HTML page @{simple_page.html} with a basic structure",
        "Generate a bash script !{ls -la} to list files",
        "Create a complex @{complex.py} example with !{echo Hello} mixed commands"
    ]

    for input_text in user_inputs:
        print(f"\nOriginal input: '{input_text}'")
        cleaned, commands = mcp.extract_mcp_from_user_input(input_text)
        print(f"Cleaned input: '{cleaned}'")
        print(f"Extracted commands: {commands}")

    # Test MCP blocks extraction
    print("\n2. Testing MCP blocks extraction from model response")

    test_responses = [
        """Here's a simple Python function:

>>> FILE: hello.py
def hello():
    print("Hello, world!")

if __name__ == "__main__":
    hello()
<

You can run it with `python hello.py`.""",

        """I'll create two files for you:

>>> FILE: index.html
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Hello World</h1>
    <p>This is a test page.</p>
</body>
</html>
<

And the CSS file:

>>> FILE: style.css
body {
    font-family: Arial, sans-serif;
    margin: 20px;
}

h1 {
    color: blue;
}
<

These files create a simple web page with a blue heading."""
    ]

    for i, response in enumerate(test_responses):
        print(f"\nTest response #{i+1}:")
        print("-" * 40)
        print(response)
        print("-" * 40)

        cleaned, mcp_blocks = mcp.extract_mcp_blocks(response)

        print("\nCleaned output:")
        print("-" * 40)
        print(cleaned)
        print("-" * 40)

        print("Extracted MCP blocks:")
        for filename, content in mcp_blocks.items():
            print(f"\nFile: {filename}")
            print("-" * 20)
            print(content)
            print("-" * 20)

    # Test saving to files
    print("\n3. Testing saving MCP blocks to files")

    test_blocks = {
        "test_python.py": "print('Hello from MCP test!')",
        "test_data.json": '{"name": "MCP Test", "status": "working"}'
    }

    results = mcp.save_mcp_blocks(test_blocks)

    for filename, success in results.items():
        full_path = os.path.join(output_dir, filename)
        if success and os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()
            print(f"Successfully saved {filename}:")
            print(f"Content: {content[:50]}{'...' if len(content) > 50 else ''}")
        else:
            print(f"Failed to save {filename}")

    # Test streaming token handling
    print("\n4. Testing streaming token handling")

    tokens = [
        "Here's", " a", " Python", " function", ":", "\n\n",
        ">", ">", ">", " ", "F", "I", "L", "E", ":", " ", "s", "t", "r", "e", "a", "m", ".", "p", "y", "\n",
        "d", "e", "f", " ", "g", "r", "e", "e", "t", "(", ")", ":", "\n",
        " ", " ", " ", " ", "p", "r", "i", "n", "t", "(", '"', "H", "e", "l", "l", "o", " ", "s", "t", "r", "e", "a", "m", "!", '"', ")", "\n",
        "<", "<", "<", "\n\n",
        "You", " can", " use", " this", " to", " test", " streaming", "."
    ]

    complete_text = ""
    displayed_text = ""
    mcp_buffer = ""

    print("\nSimulating token streaming:")
    for token in tokens:
        complete_text += token
        display_token, mcp_buffer = mcp.process_streaming_token(token, mcp_buffer)
        displayed_text += display_token

        # Show progress
        if display_token:
            print(display_token, end="", flush=True)

    print("\n\nStreaming complete.")
    print(f"MCP buffer: '{mcp_buffer}'")

    # Finalize streaming to process any MCP blocks
    finalized = mcp.finalize_streaming(complete_text)

    print("\nFinalized text:")
    print("-" * 40)
    print(finalized)
    print("-" * 40)

    # Check if the streamed file was created
    stream_file_path = os.path.join(output_dir, "stream.py")
    if os.path.exists(stream_file_path):
        with open(stream_file_path, 'r') as f:
            content = f.read()
        print(f"\nStreamed file content:")
        print("-" * 20)
        print(content)
        print("-" * 20)

    print("\nTest complete")

if __name__ == "__main__":
    test_mcp()