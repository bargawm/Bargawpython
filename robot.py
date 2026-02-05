# BARGAW M MY AI ROBOT
"""
NexusFile-CLI: LLM-Powered Secure File Management System
Author: Senior Software Architect
Version: 1.0.0
License: MIT

A professional CLI tool that leverages Gemini 1.5 Pro via OpenRouter to manage
Windows file systems using natural language with mandatory human-in-the-loop verification.
"""

import os
import sys
import json
import shutil
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import time


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

VERSION = "1.0.0"
MODEL_NAME = "google/gemini-pro-1.5"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ANSI Color Codes for Professional UI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class FileOperation:
    """Represents a single file operation for human verification."""
    operation_type: str  # 'search', 'move', 'copy', 'delete'
    source: str
    destination: Optional[str] = None
    details: Optional[str] = None
    
    def to_row(self) -> List[str]:
        """Convert to table row format."""
        return [
            self.operation_type.upper(),
            self.source,
            self.destination or "N/A",
            self.details or ""
        ]


@dataclass
class ToolResult:
    """Standardized result from tool execution."""
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_banner() -> None:
    """Display application banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                    NexusFile-CLI v{VERSION}                    ║
║           LLM-Powered Secure File Management                 ║
║              [Human-in-the-Loop Verification]                ║
╚══════════════════════════════════════════════════════════════╝
{Colors.ENDC}
    """
    print(banner)


def print_error(message: str) -> None:
    """Print formatted error message."""
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")


def print_success(message: str) -> None:
    """Print formatted success message."""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {message}")


def print_warning(message: str) -> None:
    """Print formatted warning message."""
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")


def print_info(message: str) -> None:
    """Print formatted info message."""
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {message}")


def check_api_key() -> str:
    """
    Verify OPENROUTER_API_KEY environment variable exists.
    
    Returns:
        str: The API key value
        
    Raises:
        SystemExit: If API key is not found
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print_error("OPENROUTER_API_KEY environment variable not found!")
        print(f"""
{Colors.CYAN}Setup Guide:{Colors.ENDC}
1. Visit https://openrouter.ai/ and create an account
2. Generate an API key from your dashboard
3. Set the environment variable:

   {Colors.BOLD}Windows PowerShell:{Colors.ENDC}
   $env:OPENROUTER_API_KEY="your-api-key-here"

   {Colors.BOLD}Windows CMD:{Colors.ENDC}
   set OPENROUTER_API_KEY=your-api-key-here

   {Colors.BOLD}Permanent (System Properties):{Colors.ENDC}
   System Properties → Environment Variables → New System Variable
        """)
        sys.exit(1)
    
    return api_key


def format_table(headers: List[str], rows: List[List[str]], max_width: int = 80) -> str:
    """
    Create a formatted ASCII table for CLI display.
    
    Args:
        headers: Column headers
        rows: Data rows
        max_width: Maximum width for truncation
        
    Returns:
        str: Formatted table string
    """
    if not rows:
        return "No data to display."
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Cap widths to prevent overflow
    col_widths = [min(w, max_width // len(headers)) for w in col_widths]
    
    # Build separator
    separator = "+".join(["-" * (w + 2) for w in col_widths])
    separator = f"+{separator}+"
    
    # Build header
    header_cells = [f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)]
    header_row = f"|{Colors.BOLD}{'|'.join(header_cells)}{Colors.ENDC}|"
    
    # Build rows
    data_rows = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cell_str = str(cell)[:col_widths[i]]
            cells.append(f" {cell_str:<{col_widths[i]}} ")
        data_rows.append(f"|{'|'.join(cells)}|")
    
    # Assemble table
    table = [
        separator,
        header_row,
        separator,
        *data_rows,
        separator
    ]
    
    return "\n".join(table)


def human_verification(operations: List[FileOperation]) -> bool:
    """
    Display proposed operations and require explicit [Y/N] confirmation.
    
    Args:
        operations: List of file operations to display
        
    Returns:
        bool: True if user confirms, False otherwise
    """
    if not operations:
        print_warning("No operations to verify.")
        return False
    
    print(f"\n{Colors.WARNING}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗")
    print("║           HUMAN VERIFICATION REQUIRED                        ║")
    print("║     Review the following operations before execution:        ║")
    print("╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}\n")
    
    headers = ["Operation", "Source", "Destination", "Details"]
    rows = [op.to_row() for op in operations]
    
    print(format_table(headers, rows))
    
    print(f"\n{Colors.CYAN}Total operations: {len(operations)}{Colors.ENDC}")
    print(f"{Colors.WARNING}This action may modify your file system.{Colors.ENDC}\n")
    
    while True:
        try:
            response = input(
                f"{Colors.BOLD}Do you want to proceed? [Y/N]: {Colors.ENDC}"
            ).strip().upper()
            
            if response == 'Y':
                return True
            elif response == 'N':
                print_info("Operation cancelled by user.")
                return False
            else:
                print_warning("Please enter 'Y' for yes or 'N' for no.")
        except (KeyboardInterrupt, EOFError):
            print("\n")
            print_info("Operation cancelled.")
            return False


# =============================================================================
# FILE OPERATION TOOLS
# =============================================================================

class FileTools:
    """Collection of file system operations with safety checks."""
    
    @staticmethod
    def search_files(
        directory: str,
        extension: Optional[str] = None,
        search_term: Optional[str] = None
    ) -> ToolResult:
        """
        Recursively search for files in a directory.
        
        Args:
            directory: Root directory to search
            extension: File extension to filter (e.g., '.txt', '.log')
            search_term: Term to search in filename
            
        Returns:
            ToolResult: Search results or error
        """
        try:
            root_path = Path(directory).expanduser().resolve()
            
            if not root_path.exists():
                return ToolResult(
                    success=False,
                    message=f"Directory not found: {directory}",
                    error="Path does not exist"
                )
            
            if not root_path.is_dir():
                return ToolResult(
                    success=False,
                    message=f"Path is not a directory: {directory}",
                    error="Invalid directory"
                )
            
            matches = []
            
            # Normalize extension
            if extension and not extension.startswith('.'):
                extension = f'.{extension}'
            
            # Recursive search
            for file_path in root_path.rglob('*'):
                if not file_path.is_file():
                    continue
                
                # Check extension filter
                if extension and file_path.suffix.lower() != extension.lower():
                    continue
                
                # Check search term in filename
                if search_term and search_term.lower() not in file_path.name.lower():
                    continue
                
                matches.append(str(file_path))
            
            return ToolResult(
                success=True,
                message=f"Found {len(matches)} file(s)",
                data=matches
            )
            
        except PermissionError as e:
            return ToolResult(
                success=False,
                message=f"Permission denied accessing {directory}",
                error=str(e)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Search failed: {str(e)}",
                error=str(e)
            )
    
    @staticmethod
    def move_files(
        file_list: List[str],
        destination_folder: str
    ) -> ToolResult:
        """
        Safely move files to destination with automatic directory creation.
        
        Args:
            file_list: List of file paths to move
            destination_folder: Target directory
            
        Returns:
            ToolResult: Operation results
        """
        try:
            dest_path = Path(destination_folder).expanduser().resolve()
            
            # Create destination if needed
            dest_path.mkdir(parents=True, exist_ok=True)
            
            if not dest_path.is_dir():
                return ToolResult(
                    success=False,
                    message=f"Destination is not a directory: {destination_folder}",
                    error="Invalid destination"
                )
            
            moved_files = []
            failed_files = []
            
            for file_str in file_list:
                try:
                    src_path = Path(file_str).expanduser().resolve()
                    
                    if not src_path.exists():
                        failed_files.append((file_str, "Source file not found"))
                        continue
                    
                    if not src_path.is_file():
                        failed_files.append((file_str, "Path is not a file"))
                        continue
                    
                    # Handle filename conflicts
                    dest_file = dest_path / src_path.name
                    counter = 1
                    original_dest = dest_file
                    while dest_file.exists():
                        stem = original_dest.stem
                        suffix = original_dest.suffix
                        dest_file = dest_path / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    shutil.move(str(src_path), str(dest_file))
                    moved_files.append(str(dest_file))
                    
                except PermissionError:
                    failed_files.append((file_str, "Permission denied"))
                except Exception as e:
                    failed_files.append((file_str, str(e)))
            
            # Prepare result message
            msg = f"Successfully moved {len(moved_files)} file(s) to {dest_path}"
            if failed_files:
                msg += f"\nFailed to move {len(failed_files)} file(s)"
            
            return ToolResult(
                success=len(failed_files) == 0,
                message=msg,
                data={"moved": moved_files, "failed": failed_files}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Move operation failed: {str(e)}",
                error=str(e)
            )
    
    @staticmethod
    def copy_files(
        file_list: List[str],
        destination_folder: str
    ) -> ToolResult:
        """
        Safely copy files to destination with automatic directory creation.
        
        Args:
            file_list: List of file paths to copy
            destination_folder: Target directory
            
        Returns:
            ToolResult: Operation results
        """
        try:
            dest_path = Path(destination_folder).expanduser().resolve()
            dest_path.mkdir(parents=True, exist_ok=True)
            
            copied_files = []
            failed_files = []
            
            for file_str in file_list:
                try:
                    src_path = Path(file_str).expanduser().resolve()
                    
                    if not src_path.exists() or not src_path.is_file():
                        failed_files.append((file_str, "Source file not found"))
                        continue
                    
                    # Handle filename conflicts
                    dest_file = dest_path / src_path.name
                    counter = 1
                    original_dest = dest_file
                    while dest_file.exists():
                        stem = original_dest.stem
                        suffix = original_dest.suffix
                        dest_file = dest_path / f"{stem}_copy_{counter}{suffix}"
                        counter += 1
                    
                    shutil.copy2(str(src_path), str(dest_file))
                    copied_files.append(str(dest_file))
                    
                except Exception as e:
                    failed_files.append((file_str, str(e)))
            
            return ToolResult(
                success=len(failed_files) == 0,
                message=f"Copied {len(copied_files)} file(s)",
                data={"copied": copied_files, "failed": failed_files}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Copy operation failed: {str(e)}",
                error=str(e)
            )
    
    @staticmethod
    def delete_files(file_list: List[str]) -> ToolResult:
        """
        Permanently delete files (with extra safety warnings).
        
        Args:
            file_list: List of file paths to delete
            
        Returns:
            ToolResult: Operation results
        """
        deleted_files = []
        failed_files = []
        
        for file_str in file_list:
            try:
                src_path = Path(file_str).expanduser().resolve()
                
                if not src_path.exists():
                    failed_files.append((file_str, "File not found"))
                    continue
                
                if not src_path.is_file():
                    failed_files.append((file_str, "Path is not a file"))
                    continue
                
                os.remove(str(src_path))
                deleted_files.append(file_str)
                
            except Exception as e:
                failed_files.append((file_str, str(e)))
        
        return ToolResult(
            success=len(failed_files) == 0,
            message=f"Deleted {len(deleted_files)} file(s)",
            data={"deleted": deleted_files, "failed": failed_files}
        )


# =============================================================================
# TOOL SCHEMAS FOR GEMINI API
# =============================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Recursively search for files in a directory with optional filtering by extension or filename pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Root directory to search (supports ~ for home directory)"
                    },
                    "extension": {
                        "type": "string",
                        "description": "File extension to filter by (e.g., 'txt', 'log', 'pdf'). Include or exclude the dot."
                    },
                    "search_term": {
                        "type": "string",
                        "description": "Optional search term to match in filenames (case-insensitive)"
                    }
                },
                "required": ["directory"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_files",
            "description": "Move files from source locations to a destination folder. Creates destination if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of absolute file paths to move"
                    },
                    "destination_folder": {
                        "type": "string",
                        "description": "Destination directory path (will be created if it doesn't exist)"
                    }
                },
                "required": ["file_list", "destination_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "copy_files",
            "description": "Copy files from source locations to a destination folder. Creates destination if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of absolute file paths to copy"
                    },
                    "destination_folder": {
                        "type": "string",
                        "description": "Destination directory path (will be created if it doesn't exist)"
                    }
                },
                "required": ["file_list", "destination_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_files",
            "description": "Permanently delete files. Use with caution - files are not sent to recycle bin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of absolute file paths to delete permanently"
                    }
                },
                "required": ["file_list"]
            }
        }
    }
]


# =================            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name
        }
        
        system_msg = """
You are a File System Assistant. You respond ONLY with Python code to solve the user's request.
You must use the provided `FileSystemTools` class.

AVAILABLE TOOLS:
1. FileSystemTools.find_files(pattern, root_path=".", recursive=True) -> returns list of strings (paths)
2. FileSystemTools.move_files(file_paths, destination) -> moves files

RULES:
- Return ONLY the Python code block.
- Use FileSystemTools methods.
- No 'if __name__ == "__main__":' blocks.
- Use forward slashes (/) for all paths.
- Wrap code in ```python ... ``` tags.
"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                print(f"{Fore.RED}[Error] API Request failed: {resp.status_code}")
                print(f"{Fore.RED}Response text: {resp.text}")
                return None
            content = resp.json()["choices"][0]["message"]["content"]
            if "```python" in content:
                return content.split("```python")[1].split("```")[0].strip()
            return content.strip()
        except Exception as e:
            print(f"{Fore.RED}[Error] API Request failed: {e}")
            return None

    def execute_command(self, user_prompt: str):
        print(f"{Fore.CYAN}Processing: {user_prompt}")
        code = self._get_code_from_ai(user_prompt)
        
        if not code:
            return
            
        print(f"\n{Fore.GREEN}--- PROPOSED PLAN ---{Style.RESET_ALL}")
        print(code)
        print(f"{Fore.GREEN}---------------------{Style.RESET_ALL}\n")
        
        confirm = input(f"{Fore.YELLOW}Execute this plan? (y/N): ").strip().lower()
        if confirm == 'y':
            try:
                # Provide tools and helpers to the execution environment
                context = {
                    "FileSystemTools": FileSystemTools,
                    "os": os,
                    "shutil": shutil,
                    "Path": Path,
                    "print": print
                }
                exec(code, context)
                print(f"{Fore.GREEN}All tasks finished.")
            except Exception as e:
                print(f"{Fore.RED}[Error] Execution failed: {e}")
        else:
            print(f"{Fore.RED}Execution cancelled by user.")

def main():
    parser = argparse.ArgumentParser(description="Gemini Robot CLI - Restarted Version")
    parser.add_argument("prompt", nargs="?", help="Natural language command")
    args = parser.parse_args()
    
    robot = GeminiRobot()
    if args.prompt:
        robot.execute_command(args.prompt)
    else:
        print(f"{Fore.YELLOW}Usage: python robot.py \"Your command here\"")

if __name__ == "__main__":
    main()
