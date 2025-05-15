"""Utilities for file operations and displaying HTML."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def open_in_browser(html_content: str, temp_file_prefix: str = "prompt_attribution_") -> str:
    """Open HTML content in the default web browser.
    
    Args:
        html_content: HTML content to display
        temp_file_prefix: Prefix for the temporary file
        
    Returns:
        Path to the temporary file
    """
    # Create a temporary file with the HTML content
    fd, path = tempfile.mkstemp(suffix=".html", prefix=temp_file_prefix)
    
    try:
        with os.fdopen(fd, "w") as f:
            f.write(html_content)
        
        # Convert to a file:// URL
        url = f"file://{Path(path).resolve()}"
        
        # Open the URL in the default browser
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", url], check=True)
        elif sys.platform == "win32":  # Windows
            subprocess.run(["start", url], shell=True, check=True)
        else:  # Linux and others
            subprocess.run(["xdg-open", url], check=True)
            
        return path
    except Exception as e:
        print(f"Error opening browser: {e}")
        print(f"HTML written to: {path}")
        return path


def save_visualization(html_content: str, filepath: Optional[str] = None) -> str:
    """Save HTML content to a file and optionally open it.
    
    Args:
        html_content: HTML content to save
        filepath: Path to save the file, or None to use a temp file
        
    Returns:
        Path to the saved file
    """
    if filepath:
        with open(filepath, "w") as f:
            f.write(html_content)
        return filepath
    else:
        return open_in_browser(html_content) 