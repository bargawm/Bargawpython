import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama for Windows terminal support
init(autoreset=True)

# Load context from .env
load_dotenv()

class FileSystemTools:
    """Standardized Python tools for file operations."""
    
    @staticmethod
    def find_files(pattern: str, root_path: str = ".", recursive: bool = True) -> List[str]:
        """Search for files matching a pattern."""
        print(f"{Fore.CYAN}Searching for '{pattern}' in '{root_path}'...")
        found = []
        try:
            root = Path(root_path)
            if not root.exists():
                print(f"{Fore.RED}[Error] Path not found: {root_path}")
                return []
            
            glob_func = root.rglob if recursive else root.glob
            for p in glob_func(pattern):
                if p.is_file():
                    found.append(str(p.absolute()))
            print(f"{Fore.GREEN}Found {len(found)} file(s).")
        except Exception as e:
            print(f"{Fore.RED}[Error] Search failed: {e}")
        return found

    @staticmethod
    def move_files(file_paths: List[str], destination: str):
        """Move a list of files to a destination folder."""
        print(f"{Fore.CYAN}Moving {len(file_paths)} files to '{destination}'...")
        dest_path = Path(destination)
        dest_path.mkdir(parents=True, exist_ok=True)
        success_count = 0
        for path_str in file_paths:
            try:
                src = Path(path_str)
                if src.exists():
                    dst = dest_path / src.name
                    shutil.move(str(src), str(dst))
                    print(f"{Fore.GREEN}Moved: {src.name}")
                    success_count += 1
            except Exception as e:
                print(f"{Fore.RED}[Error] Could not move {path_str}: {e}")
        print(f"{Fore.YELLOW}Operation complete: {success_count}/{len(file_paths)} moved.")

class GeminiRobot:
    """The main CLI Robot agent."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("LLM_MODEL", "stepfun/step-3.5-flash:free")
        self.site_url = os.getenv("site_url", "https://github.com/Unknownuser1000989/gemini-robot-cli")
        self.site_name = os.getenv("site_name", "Gemini Robot CLI")
        
        if not self.api_key or self.api_key == "your_api_key_here":
            print(f"{Fore.RED}[Error] Missing OPENROUTER_API_KEY in .env file.")
            sys.exit(1)

    def _get_code_from_ai(self, user_prompt: str) -> Optional[str]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
