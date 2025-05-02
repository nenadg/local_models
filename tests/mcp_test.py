"""
MCP (Model Content Protocol) Standalone Test

This script tests the Model Content Protocol functionality for piping LLM outputs to files.
It simulates a chat session and demonstrates how MCP commands can save content to files.

Usage:
    python test_mcp.py
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mcp_handler import MCPHandler
except ImportError as e:
    print(f"Error importing MCP handler: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def test_mcp_commands():
    """Test MCP user command handling"""
    print("\n===== Testing MCP User Commands =====")
    
    # Initialize MCP handler
    output_dir = "./mcp_output"
    os.makedirs(output_dir, exist_ok=True)
    
    mcp = MCPHandler(output_dir=output_dir, allow_shell_commands=True)
    
    # Test various user command types
    test_commands = [
        "Create a Python script @{hello.py}",
        "Generate HTML @{index.html} for a simple website",
        "List all files in current directory !{ls -la}",
        "Make a Markdown document @{notes.md} about programming",
        "Get system info !{uname -a} and save it @{system_info.txt}"
    ]
    
    for cmd in test_commands:
        print(f"\nOriginal: {cmd}")
        cleaned, commands = mcp.extract_mcp_from_user_input(cmd)
        print(f"Cleaned: {cleaned}")
        print(f"Commands: {commands}")
    
    print("\nUser command tests completed.")


def test_mcp_blocks():
    """Test MCP block extraction from model responses"""
    print("\n===== Testing MCP Block Extraction =====")
    
    # Initialize MCP handler
    output_dir = "./mcp_output"
    os.makedirs(output_dir, exist_ok=True)
    
    mcp = MCPHandler(output_dir=output_dir, allow_shell_commands=True)
    
    # Create test responses with MCP blocks
    test_responses = [
        # Python script example
        """Here's a simple Python function to calculate the factorial:

>>> FILE: factorial.py
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

if __name__ == "__main__":
    num = 5
    print(f"Factorial of {num} is {factorial(num)}")
<<<

You can run this script with `python factorial.py`.""",

        # Multiple file example
        """Let's create a simple web page with HTML and CSS:

>>> FILE: simple.html
<!DOCTYPE html>
<html>
<head>
    <title>Simple Page</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Hello World</h1>
    <p>This is a simple web page.</p>
</body>
</html>
<<<

And here's the CSS file:

>>> FILE: style.css
body {
    font-family: Arial, sans-serif;
    margin: 40px;
    line-height: 1.6;
}

h1 {
    color: #0066cc;
}

p {
    color: #333;
}
<<<

You can open the HTML file in any web browser.""",

        # Markdown example
        """Here's a Markdown document with some formatting examples:

>>> FILE: markdown_guide.md
# Markdown Guide

## Headers

# H1
## H2
### H3

## Text Formatting

*Italic text* or _italic text_
**Bold text** or __bold text__
***Bold and italic***

## Lists

### Ordered List
1. First item
2. Second item
3. Third item

### Unordered List
- Item 1
- Item 2
- Item 3

## Code

Inline `code` example

```python
def hello():
    print("Hello, world!")
```

## Links

[Markdown Guide](https://www.markdownguide.org/)
<<<

This covers the basics of Markdown syntax."""
    ]
    
    # Process each test response
    for i, response in enumerate(test_responses):
        print(f"\n--- Test Response #{i+1} ---")
        
        # Extract MCP blocks
        cleaned, blocks = mcp.extract_mcp_blocks(response)
        
        print(f"Found {len(blocks)} MCP blocks:")
        for filename, content in blocks.items():
            print(f"  - {filename}: {len(content)} characters")
        
        # Save blocks to files
        results = mcp.save_mcp_blocks(blocks)
        
        for filename, success in results.items():
            filepath = os.path.join(output_dir, filename)
            status = "Saved successfully" if success and os.path.exists(filepath) else "Failed to save"
            filesize = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            print(f"  - {filename}: {status} ({filesize} bytes)")
            
        print(f"\nCleaned text ({len(cleaned)} chars):")
        print("-" * 40)
        print(cleaned[:100] + "..." if len(cleaned) > 100 else cleaned)
        print("-" * 40)
    
    print("\nMCP block tests completed. Files saved to:", os.path.abspath(output_dir))


def test_streaming_mcp():
    """Test MCP handling during streaming token generation"""
    print("\n===== Testing Streaming MCP Handling =====")
    
    # Initialize MCP handler
    output_dir = "./mcp_output"
    os.makedirs(output_dir, exist_ok=True)
    
    mcp = MCPHandler(output_dir=output_dir, allow_shell_commands=True)
    
    # Define a streaming scenario
    print("\nSimulating token streaming with MCP content...")
    
    tokens = [
        "Here's", " a", " JavaScript", " function", " that", " calculates", " the", " sum", " of", " an", " array", ":", "\n\n",
        ">", ">", ">", " ", "F", "I", "L", "E", ":", " ", "s", "u", "m", ".", "j", "s", "\n",
        "f", "u", "n", "c", "t", "i", "o", "n", " ", "s", "u", "m", "A", "r", "r", "a", "y", "(", "a", "r", "r", ")", " ", "{", "\n",
        " ", " ", "r", "e", "t", "u", "r", "n", " ", "a", "r", "r", ".", "r", "e", "d", "u", "c", "e", "(", "(", "a", ",", " ", "b", ")", " ", "=", ">", " ", "a", " ", "+", " ", "b", ",", " ", "0", ")", ";", "\n",
        "}", "\n\n",
        "c", "o", "n", "s", "t", " ", "n", "u", "m", "b", "e", "r", "s", " ", "=", " ", "[", "1", ",", " ", "2", ",", " ", "3", ",", " ", "4", ",", " ", "5", "]", ";", "\n",
        "c", "o", "n", "s", "o", "l", "e", ".", "l", "o", "g", "(", "`", "S", "u", "m", ":", " ", "${", "s", "u", "m", "A", "r", "r", "a", "y", "(", "n", "u", "m", "b", "e", "r", "s", ")", "}", "`", ")", ";", "\n",
        "<", "<", "<", "\n\n",
        "You", " can", " run", " this", " with", " Node.js", " by", " typing", " `node", " sum.js`", "."
    ]
    
    # Process token stream
    complete_text = ""
    displayed_text = ""
    mcp_buffer = ""
    
    print("\nDisplayed tokens during streaming:")
    print("-" * 40)
    
    for token in tokens:
        # Add to complete text
        complete_text += token
        
        # Process through MCP handler
        display_token, mcp_buffer = mcp.process_streaming_token(token, mcp_buffer)
        
        # Add displayable tokens to output
        if display_token:
            displayed_text += display_token
            print(display_token, end="", flush=True)
            
        # Simulate processing delay
        time.sleep(0.01)
    
    print("\n" + "-" * 40)
    
    # Finalize streaming
    finalized = mcp.finalize_streaming(complete_text)
    
    print(f"\nFinalized visible text ({len(finalized)} chars):")
    print("-" * 40)
    print(finalized)
    print("-" * 40)
    
    # Verify file was created
    mcp_file = os.path.join(output_dir, "sum.js")
    if os.path.exists(mcp_file):
        with open(mcp_file, 'r') as f:
            content = f.read()
        print(f"\nCreated file: {mcp_file}")
        print("-" * 40)
        print(content)
        print("-" * 40)
        print(f"File size: {os.path.getsize(mcp_file)} bytes")
    else:
        print(f"\nError: Expected file {mcp_file} was not created")
    
    print("\nStreaming MCP test completed.")


def test_save_response():
    """Test saving a complete response to a file"""
    print("\n===== Testing Save Response Functionality =====")
    
    # Initialize MCP handler
    output_dir = "./mcp_output"
    os.makedirs(output_dir, exist_ok=True)
    
    mcp = MCPHandler(output_dir=output_dir, allow_shell_commands=True)
    
    # Create a test response
    test_response = """# Python Best Practices

## Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Limit line length to 79 characters

## Error Handling
- Use try/except blocks appropriately
- Catch specific exceptions, not Exception
- Handle errors gracefully

## Documentation
- Write docstrings for all functions and classes
- Include examples in documentation
- Keep comments up-to-date

## Testing
- Write unit tests for all code
- Use pytest for testing
- Aim for high code coverage
"""
    
    # Save the response to a file
    filename = "python_best_practices.md"
    success = mcp.save_response_to_file(test_response, filename)
    
    filepath = os.path.join(output_dir, filename)
    if success and os.path.exists(filepath):
        print(f"Successfully saved response to {filepath}")
        print(f"File size: {os.path.getsize(filepath)} bytes")
        
        # Show first few lines
        with open(filepath, 'r') as f:
            print("\nFile preview:")
            print("-" * 40)
            print("\n".join(f.readlines()[:5]))
            print("...")
            print("-" * 40)
    else:
        print(f"Failed to save response to {filepath}")
    
    print("\nSave response test completed.")


def run_integrated_test():
    """Run an integrated test simulating interaction between a model and user"""
    print("\n===== Running Integrated MCP Test =====")
    
    # Initialize MCP handler
    output_dir = "./mcp_output"
    os.makedirs(output_dir, exist_ok=True)
    
    mcp = MCPHandler(output_dir=output_dir, allow_shell_commands=True)
    
    # Simulate a conversation with MCP commands
    conversations = [
        # Conversation 1: Python script request
        {
            "user": "Write me a Python script that generates the Fibonacci sequence @{fibonacci.py}",
            "assistant": """I'll create a Python script to generate the Fibonacci sequence for you.

>>> FILE: fibonacci.py
def fibonacci(n):
    """Generate Fibonacci sequence up to n terms."""
    fib = [0, 1]
    
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

def main():
    n = int(input("Enter the number of Fibonacci terms to generate: "))
    result = fibonacci(n)
    print(f"Fibonacci sequence ({n} terms):")
    print(result)

if __name__ == "__main__":
    main()
<<<

This Python script defines two functions:
- `fibonacci(n)`: Generates the first n terms of the Fibonacci sequence
- `main()`: Handles user input and displays the result

You can run it by opening a terminal and typing:
```
python fibonacci.py
```

The script will prompt you to enter the number of terms to generate."""
        },
        
        # Conversation 2: HTML/CSS request
        {
            "user": "Create a simple webpage with HTML and CSS that has a gradient background @{gradient_page.html}",
            "assistant": """I'll create a webpage with a gradient background for you.

>>> FILE: gradient_page.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gradient Background</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            /* Linear gradient from purple to blue */
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .container {
            text-align: center;
            padding: 2rem;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            backdrop-filter: blur(5px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        p {
            font-size: 1.2rem;
            max-width: 600px;
            line-height: 1.6;
        }
        
        .button {
            display: inline-block;
            background-color: white;
            color: #6a11cb;
            padding: 0.8rem 1.5rem;
            border-radius: 30px;
            text-decoration: none;
            font-weight: bold;
            margin-top: 1rem;
            transition: transform 0.3s ease;
        }
        
        .button:hover {
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Beautiful Gradient Background</h1>
        <p>This page features a smooth gradient background transitioning from purple to blue, creating a modern and visually appealing design.</p>
        <a href="#" class="button">Learn More</a>
    </div>
</body>
</html>
<<<

I've created an HTML file with an embedded CSS style section that creates:

1. A purple-to-blue gradient background (135-degree angle)
2. A centered container with a frosted glass effect
3. Responsive typography
4. A button with a hover effect

You can open this file directly in any web browser to see the result. The gradient creates a beautiful background that transitions from a rich purple (#6a11cb) to a bright blue (#2575fc).

Would you like me to make any modifications to this design?"""
        },
        
        # Conversation 3: Multiple files
        {
            "user": "Create a simple React component that displays a counter @{Counter.jsx} and also show me how to use it in a main app @{App.jsx}",
            "assistant": """I'll create a simple React counter component and show you how to use it in a main App component.

>>> FILE: Counter.jsx
import React, { useState } from 'react';

const Counter = ({ initialValue = 0, step = 1 }) => {
  const [count, setCount] = useState(initialValue);
  
  const increment = () => {
    setCount(prevCount => prevCount + step);
  };
  
  const decrement = () => {
    setCount(prevCount => prevCount - step);
  };
  
  const reset = () => {
    setCount(initialValue);
  };
  
  return (
    <div className="counter">
      <h2>Counter: {count}</h2>
      <div className="counter-controls">
        <button onClick={decrement}>-</button>
        <button onClick={reset}>Reset</button>
        <button onClick={increment}>+</button>
      </div>
      <p>Step size: {step}</p>
    </div>
  );
};

export default Counter;
<<<

Now, here's how you would use this Counter component in your main App:

>>> FILE: App.jsx
import React from 'react';
import Counter from './Counter';
import './App.css';

function App() {
  return (
    <div className="app">
      <h1>React Counter Example</h1>
      
      <div className="counters">
        <div className="counter-container">
          <h3>Basic Counter</h3>
          <Counter />
        </div>
        
        <div className="counter-container">
          <h3>Counter with Initial Value</h3>
          <Counter initialValue={10} />
        </div>
        
        <div className="counter-container">
          <h3>Counter with Custom Step</h3>
          <Counter step={5} />
        </div>
      </div>
    </div>
  );
}

export default App;
<<<

I've also created a simple CSS file to style these components:

>>> FILE: App.css
.app {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: Arial, sans-serif;
}

h1 {
  text-align: center;
  color: #333;
}

.counters {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  justify-content: center;
}

.counter-container {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 20px;
  width: 250px;
  background-color: #f9f9f9;
}

.counter {
  text-align: center;
}

.counter h2 {
  font-size: 24px;
  margin-bottom: 16px;
}

.counter-controls {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-bottom: 16px;
}

.counter-controls button {
  padding: 8px 16px;
  font-size: 16px;
  cursor: pointer;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
}

.counter-controls button:hover {
  background-color: #45a049;
}

.counter-controls button:nth-child(1) {
  background-color: #f44336;
}

.counter-controls button:nth-child(1):hover {
  background-color: #d32f2f;
}

.counter-controls button:nth-child(2) {
  background-color: #2196F3;
}

.counter-controls button:nth-child(2):hover {
  background-color: #1976D2;
}
<<<

To use these components in a React application:

1. Place these files in your project's `src` directory
2. Make sure you have React installed
3. The Counter component is reusable and accepts two props:
   - `initialValue`: The starting count value (default: 0)
   - `step`: The increment/decrement step size (default: 1)

The App component demonstrates three different ways to use the Counter:
1. Basic counter with default values
2. Counter with a custom initial value of 10
3. Counter with a custom step size of 5

The CSS provides a clean, responsive layout with colored buttons for the controls."""
        }
    ]
    
    # Process each conversation
    for i, convo in enumerate(conversations):
        print(f"\n--- Conversation {i+1} ---")
        
        # Process user message
        user_msg = convo["user"]
        print(f"User: {user_msg}")
        
        cleaned_user, user_commands = mcp.extract_mcp_from_user_input(user_msg)
        print(f"  Cleaned message: '{cleaned_user}'")
        print(f"  Commands: {list(user_commands.keys()) if user_commands else 'None'}")
        
        # Process assistant message
        assistant_msg = convo["assistant"]
        print(f"\nAssistant: (message length: {len(assistant_msg)} chars)")
        
        cleaned_assistant, assistant_blocks = mcp.extract_mcp_blocks(assistant_msg)
        print(f"  Extracted {len(assistant_blocks)} MCP blocks: {list(assistant_blocks.keys())}")
        
        # Save MCP blocks to files
        results = mcp.save_mcp_blocks(assistant_blocks)
        success_count = sum(1 for success in results.values() if success)
        print(f"  Saved {success_count}/{len(results)} files successfully")
        
        # If user requested saving the entire response, do that too
        for filename, cmd in user_commands.items():
            if cmd.get("action") == "save_response":
                success = mcp.save_response_to_file(assistant_msg, filename)
                print(f"  Saved complete response to {filename}: {success}")
        
        # Show created files
        print("\n  Created files:")
        for filename in list(assistant_blocks.keys()) + list(user_commands.keys()):
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"    - {filename} ({file_size} bytes)")
    
    print("\nIntegrated test completed. Files saved to:", os.path.abspath(output_dir))


def main():
    """Main function to run all MCP tests"""
    print(f"Starting MCP tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = "./mcp_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output files will be saved to: {os.path.abspath(output_dir)}")
    
    try:
        # Run all tests
        test_mcp_commands()
        test_mcp_blocks()
        test_streaming_mcp()
        test_save_response()
        run_integrated_test()
        
        print("\n===== All MCP Tests Completed =====")
        print(f"Tests finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"All output files were saved to: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"Error during tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()