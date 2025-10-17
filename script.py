#!/usr/bin/env python3
"""
Qwen3-Coder 30B CLI Assistant — Version 3.0 (fixed)
Complete, self-contained Gemini-style CLI wrapper for LM Studio local API.

Key fixes:
- REPL receives CLI args properly.
- diff command argument parsing fixed.
- plugin loading hardened (no crashes on faulty plugins).
- streaming usage guarded if backend doesn't support it.
- consistent session/save behavior.
- minor API/schema fallbacks handled.
- tab-completion works; history persists.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import requests
import time
import subprocess
import tempfile
import atexit
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import threading
import itertools

# ANSI Color codes for colorful UI
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

def colorize(text: str, color: str, bold: bool = False) -> str:
    """Add color to text with optional bold formatting."""
    # Enable ANSI colors on Windows
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass
    
    formatting = Colors.BOLD if bold else ''
    return f"{formatting}{color}{text}{Colors.RESET}"

def enable_windows_colors():
    """Enable ANSI color support on Windows."""
    if os.name == 'nt':
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            
            # Enable ANSI escape sequence processing
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            STD_OUTPUT_HANDLE = -11
            
            handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
            mode = wintypes.DWORD()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode)
            return True
        except Exception:
            return False
    return True

# Windows-compatible readline import
try:
    import readline
    HAS_READLINE = True
except ImportError:
    # On Windows, readline might not be available
    try:
        import pyreadline3 as readline
        HAS_READLINE = True
    except ImportError:
        # Fallback: create dummy readline functions
        class DummyReadline:
            def read_history_file(self, filename): pass
            def write_history_file(self, filename): pass
            def get_line_buffer(self): return ""
            def set_completer(self, fn): pass
            def parse_and_bind(self, string): pass
        readline = DummyReadline()
        HAS_READLINE = False

# Optional highlighting
try:
    from pygments import highlight
    from pygments.lexers import guess_lexer_for_filename, TextLexer
    from pygments.formatters import TerminalFormatter
    HAS_PYGMENTS = True
except Exception:
    HAS_PYGMENTS = False

HOME = os.path.expanduser("~")
CONFIG_DIR = os.path.join(HOME, ".qwen_cli")
SESSION_FILE = os.path.join(CONFIG_DIR, "session.json")
HISTORY_FILE = os.path.join(CONFIG_DIR, "history.txt")
PLUGINS_DIR = os.path.join(CONFIG_DIR, "plugins")
LOG_DIR = os.path.join(CONFIG_DIR, "logs")

DEFAULT_API_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-coder-30b"
DEFAULT_PROFILE = "cli"
DEFAULT_MAX_HISTORY = 50
MAX_CONVERSATION_TOKENS = 16000  # heuristic

# Ensure directories
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(PLUGINS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Readline history
if HAS_READLINE:
    try:
        readline.read_history_file(HISTORY_FILE)
    except Exception:
        pass
    atexit.register(lambda: readline.write_history_file(HISTORY_FILE))

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def safe_read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def confirm(prompt: str) -> bool:
    ans = input(f"{prompt} [y/N]: ").strip().lower()
    return ans in ("y", "yes")

def highlight_code(filename: str, code: str) -> str:
    if not HAS_PYGMENTS:
        return code
    try:
        lexer = guess_lexer_for_filename(filename, code)
    except Exception:
        lexer = TextLexer()
    return highlight(code, lexer, TerminalFormatter())

class QwenCLI:
    def __init__(self, api_url: str, model: str, profile: str, auto_save: bool = True, verbose: bool = False):
        self.api_url = api_url
        self.model = model
        self.profile = profile
        self.auto_save = auto_save
        self.verbose = verbose
        self.system_prompts = self._default_profiles()
        self.system_prompt = self.system_prompts.get(profile, self.system_prompts["cli"])
        self.conversation: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.history: List[Dict[str, str]] = []
        self.max_history = DEFAULT_MAX_HISTORY
        self.session_path = SESSION_FILE
        self.log_base = os.path.join(LOG_DIR, f"session_{now_ts()}")
        self.trust_execution = False
        self.retry_attempts = 3
        self.backoff_base = 1.2
        self.stream_enabled = True
        self.plugins: Dict[str, Any] = {}
        self.aliases = {
            ":r": "read",
            ":w": "write",
            ":a": "append",
            ":x": "exit",
            ":h": "help",
        }
        self._lock = threading.Lock()
        self._progress_spinner = None
        self._progress_thread = None
        self._stop_progress = threading.Event()
        self._load_persistent_session()
        self._load_plugins()
        self._register_builtin_commands()

    def _default_profiles(self) -> Dict[str, str]:
        return {
            "cli": (
                "You are a command-line AI assistant. Emulate a Linux shell. "
                "Provide concise, factual outputs suitable for terminal usage. "
                "Never add unnecessary commentary. When providing code, include runnable code blocks."
            ),
            "dev": (
                "You are a developer assistant. Provide detailed code, comments, tests, and explain design trade-offs."
            ),
            "teacher": (
                "You are an educational assistant. Explain step-by-step with clear reasoning and examples."
            ),
            "analyst": (
                "You are a data analyst assistant. Focus on data, statistical reasoning, and reproducible code snippets."
            ),
        }

    def _load_persistent_session(self) -> None:
        if os.path.exists(self.session_path):
            try:
                with open(self.session_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data.get("conversation"), list):
                    self.conversation = data["conversation"]
                if isinstance(data.get("history"), list):
                    self.history = data["history"]
                self.model = data.get("model", self.model)
                self.profile = data.get("profile", self.profile)
                self.system_prompt = self.system_prompts.get(self.profile, self.system_prompt)
            except Exception:
                # ignore corrupt session file
                pass

    def save_session(self) -> None:
        if not self.auto_save:
            return
        out = {
            "conversation": self.conversation[-(self.max_history * 2):],
            "history": self.history[-self.max_history:],
            "model": self.model,
            "profile": self.profile,
            "saved_at": now_ts(),
        }
        try:
            safe_write(self.session_path, json.dumps(out, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _load_plugins(self) -> None:
        # keep safe: ignore plugins that error
        sys.path.insert(0, PLUGINS_DIR)
        for fname in os.listdir(PLUGINS_DIR):
            if not fname.endswith(".py"):
                continue
            name = fname[:-3]
            try:
                module = __import__(name)
                if hasattr(module, "register"):
                    try:
                        module.register(self)
                        self.plugins[name] = module
                    except Exception:
                        print(f"[plugin] register() failed: {name}")
                else:
                    # add module as passive plugin
                    self.plugins[name] = module
            except Exception:
                print(f"[plugin] failed to load: {name}")

    def register_command(self, name: str, fn: Callable[[str], None]) -> None:
        setattr(self, f"cmd_{name}", fn)

    def _register_builtin_commands(self) -> None:
        def cmd_model(arg: str):
            if not arg:
                print(f"{colorize('Current model:', Colors.BRIGHT_CYAN)} {colorize(self.model, Colors.BRIGHT_GREEN)}")
                return
            self.model = arg.strip()
            print(f"{colorize('[model →', Colors.BRIGHT_BLUE)} {colorize(self.model, Colors.BRIGHT_GREEN)}{colorize(']', Colors.BRIGHT_BLUE)}")
        self.register_command("model", cmd_model)

        def cmd_mode(arg: str):
            if not arg:
                print(f"{colorize('Current profile:', Colors.BRIGHT_CYAN)} {colorize(self.profile, Colors.BRIGHT_MAGENTA)}")
                return
            if arg not in self.system_prompts:
                available = ', '.join(self.system_prompts.keys())
                print(f"{colorize('Unknown profile. Available:', Colors.BRIGHT_RED)} {colorize(available, Colors.BRIGHT_YELLOW)}")
                return
            self.profile = arg
            self.system_prompt = self.system_prompts[arg]
            if self.conversation and self.conversation[0].get("role") == "system":
                self.conversation[0]["content"] = self.system_prompt
            else:
                self.conversation.insert(0, {"role": "system", "content": self.system_prompt})
            print(f"{colorize('[profile →', Colors.BRIGHT_BLUE)} {colorize(arg, Colors.BRIGHT_MAGENTA)}{colorize(']', Colors.BRIGHT_BLUE)}")
        self.register_command("mode", cmd_mode)

    def _log_verbose(self, message: str) -> None:
        """Log verbose messages if verbose mode is enabled."""
        if self.verbose:
            print(f"{colorize('[VERBOSE]', Colors.BRIGHT_BLUE)} {colorize(message, Colors.WHITE)}")

    def _start_progress_indicator(self, message: str = "Processing") -> None:
        """Start a progress spinner in a separate thread."""
        if self._progress_thread and self._progress_thread.is_alive():
            return  # Already running
        
        self._stop_progress.clear()
        self._progress_thread = threading.Thread(target=self._progress_worker, args=(message,))
        self._progress_thread.daemon = True
        self._progress_thread.start()

    def _progress_worker(self, message: str) -> None:
        """Worker function for the progress spinner."""
        spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧']
        spinner_colors = [Colors.BRIGHT_RED, Colors.BRIGHT_YELLOW, Colors.BRIGHT_GREEN, 
                         Colors.BRIGHT_CYAN, Colors.BRIGHT_BLUE, Colors.BRIGHT_MAGENTA]
        spinner = itertools.cycle(spinner_chars)
        colors = itertools.cycle(spinner_colors)
        start_time = time.time()
        
        while not self._stop_progress.is_set():
            elapsed = time.time() - start_time
            elapsed_str = f"{elapsed:.1f}s"
            spinner_char = next(spinner)
            color = next(colors)
            colored_spinner = colorize(spinner_char, color)
            colored_message = colorize(message, Colors.BRIGHT_WHITE)
            colored_time = colorize(f"({elapsed_str})", Colors.BRIGHT_CYAN)
            print(f"\r{colored_spinner} {colored_message}... {colored_time}", end="", flush=True)
            time.sleep(0.1)
        
        # Clear the spinner line
        print("\r" + " " * (len(message) + 20) + "\r", end="", flush=True)

    def _stop_progress_indicator(self) -> None:
        """Stop the progress spinner."""
        self._stop_progress.set()
        if self._progress_thread and self._progress_thread.is_alive():
            self._progress_thread.join(timeout=0.5)

    def context_show(self, n: int = 10) -> None:
        print(colorize("--- context (last messages) ---", Colors.BRIGHT_CYAN, bold=True))
        for msg in self.conversation[-n:]:
            r = msg.get("role", "")
            c = msg.get("content", "")
            if r == "system":
                prefix = colorize("[SYSTEM]", Colors.BRIGHT_YELLOW, bold=True)
            elif r == "user":
                prefix = colorize("[USER]", Colors.BRIGHT_GREEN, bold=True)
            else:
                prefix = colorize("[ASSIST]", Colors.BRIGHT_BLUE, bold=True)
            content = c[:800] + ('...' if len(c) > 800 else '')
            print(f"{prefix}: {colorize(content, Colors.WHITE)}")
        print(colorize("--- end context ---", Colors.BRIGHT_CYAN, bold=True))

    def context_clear(self) -> None:
        self.conversation = [{"role": "system", "content": self.system_prompt}]
        print(colorize("[context cleared]", Colors.BRIGHT_GREEN))

    def context_save(self, name: str) -> None:
        path = os.path.join(CONFIG_DIR, f"context_{name}.json")
        safe_write(path, json.dumps(self.conversation, ensure_ascii=False, indent=2))
        print(f"{colorize('[context saved:', Colors.BRIGHT_GREEN)} {colorize(path, Colors.BRIGHT_CYAN)}{colorize(']', Colors.BRIGHT_GREEN)}")

    def context_load(self, name: str) -> None:
        path = os.path.join(CONFIG_DIR, f"context_{name}.json")
        if not os.path.exists(path):
            print(colorize("[no such context file]", Colors.BRIGHT_RED))
            return
        try:
            data = json.loads(safe_read(path))
            if isinstance(data, list):
                self.conversation = data
                print(f"{colorize('[context loaded:', Colors.BRIGHT_GREEN)} {colorize(path, Colors.BRIGHT_CYAN)}{colorize(']', Colors.BRIGHT_GREEN)}")
            else:
                print(colorize("[invalid context file]", Colors.BRIGHT_RED))
        except Exception as e:
            print(f"{colorize('[load error:', Colors.BRIGHT_RED)} {colorize(str(e), Colors.WHITE)}{colorize(']', Colors.BRIGHT_RED)}")

    def cmd_read(self, arg: str) -> None:
        path = arg.strip()
        if not path:
            print("Usage: read <path>")
            return
        if not os.path.exists(path):
            print("[file not found]")
            return
        try:
            txt = safe_read(path)
            if HAS_PYGMENTS:
                print(highlight_code(path, txt))
            else:
                print(txt)
        except Exception as e:
            print(f"[read error: {e}]")

    def cmd_write(self, arg: str) -> None:
        parts = arg.split(" ", 1)
        if len(parts) < 2:
            print("Usage: write <path> <content>")
            return
        path, content = parts[0], parts[1]
        try:
            safe_write(path, content)
            print(f"[written: {path}]")
        except Exception as e:
            print(f"[write error: {e}]")

    def cmd_append(self, arg: str) -> None:
        parts = arg.split(" ", 1)
        if len(parts) < 2:
            print("Usage: append <path> <content>")
            return
        path, content = parts[0], parts[1]
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n" + content)
            print(f"[appended: {path}]")
        except Exception as e:
            print(f"[append error: {e}]")

    def cmd_diff(self, arg: str) -> None:
        parts = arg.split()
        if len(parts) != 2:
            print("Usage: diff <file1> <file2>")
            return
        f1, f2 = parts
        if not os.path.exists(f1) or not os.path.exists(f2):
            print("[file not found]")
            return
        try:
            import difflib
            a = safe_read(f1).splitlines(keepends=True)
            b = safe_read(f2).splitlines(keepends=True)
            d = difflib.unified_diff(a, b, fromfile=f1, tofile=f2)
            out = "".join(d)
            if not out:
                print("[no differences]")
            else:
                print(out)
        except Exception as e:
            print(f"[diff error: {e}]")

    def execute_code_snippet(self, filename_hint: str, code: str, lang: Optional[str] = None) -> None:
        if not self.trust_execution:
            ok = confirm("Execute code snippet locally? This may run arbitrary code.")
            if not ok:
                print("[execution cancelled]")
                return
        lang = (lang or "").lower()
        if not lang:
            if filename_hint.endswith(".py") or "def " in code or "import " in code:
                lang = "python"
            elif filename_hint.endswith(".sh") or code.strip().startswith("#!"):
                lang = "bash"
            else:
                lang = "text"
        if lang == "python":
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                res = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=30)
                if res.stdout:
                    print(res.stdout)
                if res.stderr:
                    print(f"[stderr] {res.stderr}")
            except subprocess.TimeoutExpired:
                print("[execution timed out]")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        elif lang in ("bash", "sh"):
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                res = subprocess.run(["bash", tmp_path], capture_output=True, text=True, timeout=30)
                if res.stdout:
                    print(res.stdout)
                if res.stderr:
                    print(f"[stderr] {res.stderr}")
            except subprocess.TimeoutExpired:
                print("[execution timed out]")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        else:
            print("[unknown language or non-executable snippet]")

    def _api_post(self, payload: Dict[str, Any], stream: bool = False) -> Any:
        headers = {"Content-Type": "application/json"}
        last_exc = None
        
        # Increased timeouts for large models
        timeout_stream = 600 if stream else 300  # 10 minutes for streaming, 5 for non-streaming
        timeout_connect = 30  # 30 seconds for connection
        
        self._log_verbose(f"Making API request to {self.api_url}")
        self._log_verbose(f"Model: {self.model}")
        self._log_verbose(f"Stream mode: {stream}")
        self._log_verbose(f"Timeout: {timeout_stream}s")
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._log_verbose(f"Attempt {attempt}/{self.retry_attempts}")
                
                if stream:
                    with requests.post(
                        self.api_url, 
                        json=payload, 
                        headers=headers, 
                        stream=True, 
                        timeout=(timeout_connect, timeout_stream)
                    ) as r:
                        r.raise_for_status()
                        self._log_verbose("Streaming response received")
                        return r.iter_lines(decode_unicode=True)
                else:
                    r = requests.post(
                        self.api_url, 
                        json=payload, 
                        headers=headers, 
                        timeout=(timeout_connect, timeout_stream)
                    )
                    r.raise_for_status()
                    self._log_verbose("Non-streaming response received")
                    return r.json()
                    
            except requests.exceptions.RequestException as e:
                last_exc = e
                self._log_verbose(f"Request failed on attempt {attempt}: {e}")
                
                if attempt < self.retry_attempts:
                    backoff = (self.backoff_base ** attempt)
                    backoff_time = min(10, backoff)  # Cap at 10 seconds
                    self._log_verbose(f"Retrying in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
                continue
                
        if last_exc:
            self._log_verbose(f"All retry attempts failed. Last error: {last_exc}")
            raise last_exc
        else:
            raise requests.exceptions.RequestException("All retry attempts failed")

    def _compress_context(self) -> None:
        total_len = sum(len(m.get("content", "")) for m in self.conversation)
        if total_len < MAX_CONVERSATION_TOKENS:
            return
        to_summarize = self.conversation[1: max(2, len(self.conversation)//2)]
        text = "\n\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])
        prompt = f"Summarize the following conversation succinctly into a short bullet summary for context preservation:\n\n{text}"
        try:
            payload = {"model": self.model, "messages": [{"role": "system", "content": self.system_prompt},
                                                          {"role": "user", "content": prompt}],
                       "temperature": 0.1}
            res = self._api_post(payload, stream=False)
            summary = ""
            if isinstance(res, dict):
                try:
                    summary = res["choices"][0]["message"]["content"].strip()
                except Exception:
                    summary = res.get("text", "") or ""
            if summary:
                kept = self.conversation[len(to_summarize)+1:]
                self.conversation = [{"role": "system", "content": self.system_prompt},
                                     {"role": "system", "content": f"Auto-summary: {summary}"}] + kept
                print("[context auto-summarized]")
        except Exception:
            self.conversation = self.conversation[-(len(self.conversation)//2):]
            print("[context trimmed (summarize failed)]")

    def _handle_streaming_response(self, stream_iter) -> str:
        buffer = ""
        printed = False
        try:
            for line in stream_iter:
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    content = None
                    if "choices" in obj:
                        ch = obj["choices"][0]
                        if isinstance(ch, dict):
                            if "delta" in ch and isinstance(ch["delta"], dict):
                                content = ch["delta"].get("content")
                            elif "text" in ch:
                                content = ch.get("text")
                            elif "message" in ch and isinstance(ch["message"], dict):
                                content = ch["message"].get("content")
                    elif "text" in obj:
                        content = obj.get("text")
                    if content:
                        print(content, end="", flush=True)
                        buffer += content
                        printed = True
                    else:
                        print(line, end="", flush=True)
                        buffer += line
                        printed = True
                except json.JSONDecodeError:
                    print(line, end="", flush=True)
                    buffer += line
                    printed = True
            if printed:
                print()
            return buffer
        except Exception:
            return buffer

    def send_user_message(self, message: str, stream: bool = False) -> None:
        for a, r in self.aliases.items():
            if message == a or message.startswith(a + " "):
                message = message.replace(a, r, 1)

        if message.startswith("read ") or message.startswith("write ") or message.startswith("append ") \
           or message.startswith("diff ") or message.startswith("history") or message.startswith("save") \
           or message.startswith("mode ") or message.startswith("model ") or message.startswith("context ") \
           or message.startswith("help") or message == "help" \
           or message.startswith("!"):
            self._run_local_or_builtin_command(message)
            return

        self._log_verbose(f"User message: {message[:100]}{'...' if len(message) > 100 else ''}")
        self.conversation.append({"role": "user", "content": message})
        self._compress_context()

        payload = {"model": self.model, "messages": self.conversation, "temperature": 0.2}
        self._log_verbose(f"Payload prepared with {len(self.conversation)} messages")

        try:
            # Start progress indicator for non-streaming requests
            if not stream or not self.stream_enabled:
                self._start_progress_indicator("Waiting for model response")
            
            if stream and self.stream_enabled:
                try:
                    self._log_verbose("Attempting streaming request")
                    self._start_progress_indicator("Connecting to model")
                    it = self._api_post(payload, stream=True)
                    self._stop_progress_indicator()
                    self._log_verbose("Streaming started, processing response")
                    resp_text = self._handle_streaming_response(it)
                except Exception as e:
                    self._stop_progress_indicator()
                    self._log_verbose(f"Streaming failed: {e}, falling back to non-streaming")
                    # fallback to non-stream
                    self._start_progress_indicator("Waiting for model response")
                    res = self._api_post(payload, stream=False)
                    self._stop_progress_indicator()
                    resp_text = ""
                    if isinstance(res, dict):
                        resp_text = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or res.get("text", "")
                    else:
                        resp_text = str(res)
                    print(resp_text)
                self.conversation.append({"role": "assistant", "content": resp_text})
                self.history.append({"input": message, "output": resp_text, "ts": now_ts()})
                self.save_session()
                return
            else:
                self._log_verbose("Making non-streaming request")
                res = self._api_post(payload, stream=False)
                self._stop_progress_indicator()
                content = ""
                if isinstance(res, dict):
                    try:
                        content = res["choices"][0]["message"]["content"].strip()
                    except Exception:
                        content = res.get("text", "")
                else:
                    content = str(res)
                print(content)
                self.conversation.append({"role": "assistant", "content": content})
                self.history.append({"input": message, "output": content, "ts": now_ts()})
                self.save_session()
        except Exception as e:
            self._stop_progress_indicator()
            self._log_verbose(f"API request failed: {e}")
            print(f"[API error: {e}]")

    def _run_local_or_builtin_command(self, message: str) -> None:
        if message.startswith("!"):
            cmd = message[1:]
            if not cmd.strip():
                print("Usage: !<shell command>")
                return
            try:
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res.stdout:
                    print(res.stdout.strip())
                if res.stderr:
                    print(f"[stderr] {res.stderr.strip()}")
            except Exception as e:
                print(f"[exec error: {e}]")
            return

        if message.startswith("context "):
            sub = message.split(" ", 1)[1].strip()
            if sub == "show":
                self.context_show(10)
                return
            if sub == "clear":
                self.context_clear()
                return
            if sub.startswith("save "):
                name = sub.split(" ", 1)[1].strip()
                self.context_save(name)
                return
            if sub.startswith("load "):
                name = sub.split(" ", 1)[1].strip()
                self.context_load(name)
                return
            print("Usage: context [show|clear|save <name>|load <name>]")
            return

        if message.startswith("read "):
            self.cmd_read(message.split(" ", 1)[1])
            return
        if message.startswith("write "):
            rest = message.split(" ", 1)[1]
            self.cmd_write(rest)
            return
        if message.startswith("append "):
            rest = message.split(" ", 1)[1]
            self.cmd_append(rest)
            return
        if message.startswith("diff "):
            rest = message.split(" ", 1)[1] if " " in message else ""
            self.cmd_diff(rest)
            return
        if message == "history":
            for i, h in enumerate(self.history[-20:], 1):
                input_ = h.get("input", "")
                ts = h.get("ts", "")
                print(f"{i}. [{ts}] {input_}")
            return
        if message == "save":
            self.save_transcripts()
            return
        if message == "help":
            print_help()
            return
        if message.startswith("mode "):
            _, arg = message.split(" ", 1)
            getattr(self, "cmd_mode", lambda a: print("[no cmd_mode]"))(arg)
            return
        if message.startswith("model "):
            _, arg = message.split(" ", 1)
            getattr(self, "cmd_model", lambda a: print("[no cmd_model]"))(arg)
            return

    def save_transcripts(self) -> None:
        ts = now_ts()
        txt_path = self.log_base + f"_{ts}.log"
        json_path = self.log_base + f"_{ts}.json"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                for h in self.history:
                    f.write(f">>> {h.get('input')}\n{h.get('output')}\n\n")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"conversation": self.conversation, "history": self.history}, f, ensure_ascii=False, indent=2)
            print(f"[session saved: {txt_path}, {json_path}]")
        except Exception as e:
            print(f"[save error: {e}]")

    def run_noninteractive(self, command: str, stream: bool = False) -> None:
        self.send_user_message(command, stream=stream)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="qwen_cli_v3.py", 
        description="🤖 QWEN CLI v3.0 — Gemini-style Local LLM Wrapper & Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py                          Start interactive REPL
  python script.py -c "Hello world"         Single command  
  python script.py --verbose -c "Code task" With progress indicators
  python script.py --mode dev               Developer assistant mode
  python script.py --no-stream             Disable streaming responses

For full feature documentation, run: python script.py (then type 'help')
        """
    )
    p.add_argument("--api-url", default=os.getenv("QWEN_API_URL", DEFAULT_API_URL), 
                   help="API endpoint URL")
    p.add_argument("--model", default=os.getenv("QWEN_MODEL", DEFAULT_MODEL),
                   help="AI model name")
    p.add_argument("--mode", "--profile", dest="mode", default=os.getenv("QWEN_MODE", DEFAULT_PROFILE),
                   help="AI personality: cli, dev, teacher, analyst")
    p.add_argument("--clear", action="store_true", help="Clear persisted session on start")
    p.add_argument("-c", "--command", help="Run single command non-interactively and exit")
    p.add_argument("--trust", action="store_true", help="Trust and auto-execute returned code snippets without confirmation")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging and progress indicators")
    p.add_argument("--auto-save", dest="auto_save", action="store_true", help="Auto-save session state (default)")
    p.add_argument("--no-auto-save", dest="auto_save", action="store_false", help="Disable auto-save")
    p.set_defaults(auto_save=True)
    return p.parse_args()

def print_banner():
    """Display the colorful QWEN CLI banner and information."""
    banner = f"""
{colorize('╔══════════════════════════════════════════════════════════════════════════════╗', Colors.BRIGHT_CYAN, bold=True)}
{colorize('║', Colors.BRIGHT_CYAN, bold=True)}                          {colorize('🤖 QWEN CLI Assistant v3.0', Colors.BRIGHT_YELLOW, bold=True)}                          {colorize('║', Colors.BRIGHT_CYAN, bold=True)}
{colorize('║', Colors.BRIGHT_CYAN, bold=True)}                    {colorize('Gemini-style Local LLM Wrapper & Manager', Colors.BRIGHT_WHITE)}                  {colorize('║', Colors.BRIGHT_CYAN, bold=True)}
{colorize('╚══════════════════════════════════════════════════════════════════════════════╝', Colors.BRIGHT_CYAN, bold=True)}

{colorize('🚀 FEATURES:', Colors.BRIGHT_GREEN, bold=True)}
  {colorize('•', Colors.GREEN)} {colorize('Interactive REPL with conversation memory', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('Streaming & non-streaming responses', Colors.WHITE)}  
  {colorize('•', Colors.GREEN)} {colorize('Code execution with syntax highlighting', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('File operations (read/write/append/diff)', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('Context management (save/load/clear)', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('Plugin system support', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('Session persistence', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('Progress indicators for large models', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('Verbose logging & debugging', Colors.WHITE)}

{colorize('📋 QUICK COMMANDS:', Colors.BRIGHT_MAGENTA, bold=True)}
  {colorize('read <file>', Colors.CYAN)}       - {colorize('Display file contents with syntax highlighting', Colors.WHITE)}
  {colorize('write <file> <content>', Colors.CYAN)} - {colorize('Write content to file', Colors.WHITE)}
  {colorize('append <file> <content>', Colors.CYAN)} - {colorize('Append content to file', Colors.WHITE)}  
  {colorize('diff <file1> <file2>', Colors.CYAN)} - {colorize('Show differences between files', Colors.WHITE)}
  {colorize('!<command>', Colors.CYAN)}        - {colorize('Execute shell commands', Colors.WHITE)}
  {colorize('context show/clear/save/load', Colors.CYAN)} - {colorize('Manage conversation context', Colors.WHITE)}
  {colorize('history', Colors.CYAN)}          - {colorize('Show recent conversations', Colors.WHITE)}
  {colorize('save', Colors.CYAN)}             - {colorize('Save session transcripts', Colors.WHITE)}
  {colorize('mode <profile>', Colors.CYAN)}   - {colorize('Switch AI personality (cli/dev/teacher/analyst)', Colors.WHITE)}
  {colorize('model <name>', Colors.CYAN)}     - {colorize('Change AI model', Colors.WHITE)}
  {colorize('exit/quit', Colors.CYAN)}        - {colorize('Exit the application', Colors.WHITE)}

{colorize('🎯 USAGE EXAMPLES:', Colors.BRIGHT_YELLOW, bold=True)}
  {colorize('python script.py', Colors.BRIGHT_BLUE)}                          - {colorize('Start interactive REPL', Colors.WHITE)}
  {colorize('python script.py -c "Hello world"', Colors.BRIGHT_BLUE)}         - {colorize('Single command', Colors.WHITE)}
  {colorize('python script.py --verbose -c "Code task"', Colors.BRIGHT_BLUE)} - {colorize('With progress indicators', Colors.WHITE)}
  {colorize('python script.py --no-stream', Colors.BRIGHT_BLUE)}             - {colorize('Disable streaming', Colors.WHITE)}
  {colorize('python script.py --mode dev', Colors.BRIGHT_BLUE)}              - {colorize('Developer assistant mode', Colors.WHITE)}

{colorize('⚙️  PROFILES:', Colors.BRIGHT_RED, bold=True)}
  {colorize('cli', Colors.YELLOW)}      - {colorize('Command-line assistant (concise, factual)', Colors.WHITE)}
  {colorize('dev', Colors.YELLOW)}      - {colorize('Developer assistant (detailed code & explanations)', Colors.WHITE)}  
  {colorize('teacher', Colors.YELLOW)}  - {colorize('Educational assistant (step-by-step explanations)', Colors.WHITE)}
  {colorize('analyst', Colors.YELLOW)}  - {colorize('Data analyst (statistical reasoning & code)', Colors.WHITE)}

{colorize('🔧 CONFIGURATION:', Colors.BRIGHT_CYAN, bold=True)}
  {colorize('Environment Variables:', Colors.BRIGHT_WHITE, bold=True)}
    {colorize('QWEN_API_URL', Colors.GREEN)}   - {colorize('API endpoint (default: http://localhost:1234/v1/chat/completions)', Colors.WHITE)}
    {colorize('QWEN_MODEL', Colors.GREEN)}     - {colorize('Model name (default: qwen/qwen3-coder-30b)', Colors.WHITE)}
    {colorize('QWEN_MODE', Colors.GREEN)}      - {colorize('Default profile (default: cli)', Colors.WHITE)}

  {colorize('Config Directory: ~/.qwen_cli/', Colors.BRIGHT_WHITE, bold=True)}
    {colorize('session.json', Colors.GREEN)}   - {colorize('Persistent conversation state', Colors.WHITE)}
    {colorize('history.txt', Colors.GREEN)}    - {colorize('Command history for readline', Colors.WHITE)}
    {colorize('plugins/', Colors.GREEN)}       - {colorize('Custom plugin directory', Colors.WHITE)}
    {colorize('logs/', Colors.GREEN)}          - {colorize('Session transcripts and logs', Colors.WHITE)}

{colorize('📝 KEYBOARD SHORTCUTS:', Colors.BRIGHT_MAGENTA, bold=True)}
  {colorize('Tab', Colors.YELLOW)}            - {colorize('Command completion', Colors.WHITE)}
  {colorize('Ctrl+C', Colors.YELLOW)}         - {colorize('Interrupt current operation', Colors.WHITE)}
  {colorize('Ctrl+D', Colors.YELLOW)}         - {colorize('Exit (EOF)', Colors.WHITE)}
  {colorize('Up/Down arrows', Colors.YELLOW)} - {colorize('Command history navigation', Colors.WHITE)}

{colorize('💡 PRO TIPS:', Colors.BRIGHT_GREEN, bold=True)}
  {colorize('•', Colors.GREEN)} {colorize('Use --verbose to see what\'s happening with large models', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('Save important contexts with \'context save <name>\'', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('Use --no-stream for faster responses on some models', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('Prefix shell commands with ! for quick system operations', Colors.WHITE)}
  {colorize('•', Colors.GREEN)} {colorize('Type \'help\' anytime for command reference', Colors.WHITE)}

"""
    print(banner)

def print_help():
    """Display detailed colorful help information."""
    help_text = f"""
{colorize('🔍 DETAILED COMMAND REFERENCE:', Colors.BRIGHT_CYAN, bold=True)}

{colorize('📁 FILE OPERATIONS:', Colors.BRIGHT_YELLOW, bold=True)}
  {colorize('read <path>', Colors.CYAN)}                    - {colorize('Display file with syntax highlighting', Colors.WHITE)}
  {colorize('write <path> <content>', Colors.CYAN)}         - {colorize('Write new file or overwrite existing', Colors.WHITE)}
  {colorize('append <path> <content>', Colors.CYAN)}        - {colorize('Add content to end of file', Colors.WHITE)}
  {colorize('diff <file1> <file2>', Colors.CYAN)}          - {colorize('Show unified diff between files', Colors.WHITE)}

{colorize('💬 CONVERSATION MANAGEMENT:', Colors.BRIGHT_GREEN, bold=True)}
  {colorize('context show', Colors.CYAN)}                   - {colorize('Display recent conversation messages', Colors.WHITE)}
  {colorize('context clear', Colors.CYAN)}                  - {colorize('Clear conversation history', Colors.WHITE)}
  {colorize('context save <name>', Colors.CYAN)}            - {colorize('Save current context to file', Colors.WHITE)}
  {colorize('context load <name>', Colors.CYAN)}            - {colorize('Load saved context from file', Colors.WHITE)}
  {colorize('history', Colors.CYAN)}                        - {colorize('Show recent command history', Colors.WHITE)}
  {colorize('save', Colors.CYAN)}                          - {colorize('Save full session transcript', Colors.WHITE)}

{colorize('🔧 SYSTEM OPERATIONS:', Colors.BRIGHT_MAGENTA, bold=True)}
  {colorize('!<command>', Colors.CYAN)}                     - {colorize('Execute shell command (e.g., !ls, !git status)', Colors.WHITE)}
  {colorize('mode <profile>', Colors.CYAN)}                 - {colorize('Switch AI personality:', Colors.WHITE)}
    {colorize('• cli', Colors.YELLOW)}      - {colorize('Concise, factual responses', Colors.WHITE)}
    {colorize('• dev', Colors.YELLOW)}      - {colorize('Detailed code explanations', Colors.WHITE)}
    {colorize('• teacher', Colors.YELLOW)}  - {colorize('Educational step-by-step', Colors.WHITE)}
    {colorize('• analyst', Colors.YELLOW)}  - {colorize('Data-focused responses', Colors.WHITE)}
  {colorize('model <name>', Colors.CYAN)}                   - {colorize('Change AI model', Colors.WHITE)}

{colorize('🎛️  COMMAND LINE OPTIONS:', Colors.BRIGHT_RED, bold=True)}
  {colorize('-c, --command <text>', Colors.CYAN)}           - {colorize('Run single command and exit', Colors.WHITE)}
  {colorize('-v, --verbose', Colors.CYAN)}                  - {colorize('Enable detailed logging', Colors.WHITE)}
  {colorize('--no-stream', Colors.CYAN)}                    - {colorize('Disable streaming responses', Colors.WHITE)}
  {colorize('--api-url <url>', Colors.CYAN)}                - {colorize('Custom API endpoint', Colors.WHITE)}
  {colorize('--model <name>', Colors.CYAN)}                 - {colorize('Specify AI model', Colors.WHITE)}
  {colorize('--mode <profile>', Colors.CYAN)}               - {colorize('Set initial personality', Colors.WHITE)}
  {colorize('--clear', Colors.CYAN)}                        - {colorize('Clear saved session on start', Colors.WHITE)}
  {colorize('--trust', Colors.CYAN)}                        - {colorize('Auto-execute code without confirmation', Colors.WHITE)}
  {colorize('--no-auto-save', Colors.CYAN)}                - {colorize('Disable automatic session saving', Colors.WHITE)}

{colorize('🔄 ALIASES (shortcuts):', Colors.BRIGHT_BLUE, bold=True)}
  {colorize(':r', Colors.YELLOW)}  → {colorize('read', Colors.CYAN)}        {colorize(':w', Colors.YELLOW)}  → {colorize('write', Colors.CYAN)}       {colorize(':a', Colors.YELLOW)}  → {colorize('append', Colors.CYAN)}
  {colorize(':x', Colors.YELLOW)}  → {colorize('exit', Colors.CYAN)}        {colorize(':h', Colors.YELLOW)}  → {colorize('help', Colors.CYAN)}

{colorize('📋 EXAMPLE WORKFLOWS:', Colors.BRIGHT_WHITE, bold=True)}

  {colorize('🎯 Code Development:', Colors.BRIGHT_GREEN, bold=True)}
    {colorize('>>> mode dev', Colors.BRIGHT_BLUE)}
    {colorize('>>> Write a Python class for user authentication', Colors.BRIGHT_BLUE)}
    {colorize('>>> read auth.py', Colors.BRIGHT_BLUE)}
    {colorize('>>> context save auth_discussion', Colors.BRIGHT_BLUE)}

  {colorize('📊 Data Analysis:', Colors.BRIGHT_YELLOW, bold=True)}
    {colorize('>>> mode analyst', Colors.BRIGHT_BLUE)}  
    {colorize('>>> Load CSV and analyze sales trends', Colors.BRIGHT_BLUE)}
    {colorize('>>> !ls *.csv', Colors.BRIGHT_BLUE)}
    {colorize('>>> read sales_data.csv', Colors.BRIGHT_BLUE)}

  {colorize('🎓 Learning:', Colors.BRIGHT_MAGENTA, bold=True)}
    {colorize('>>> mode teacher', Colors.BRIGHT_BLUE)}
    {colorize('>>> Explain how neural networks work', Colors.BRIGHT_BLUE)}
    {colorize('>>> context save ml_learning', Colors.BRIGHT_BLUE)}

  {colorize('💻 System Admin:', Colors.BRIGHT_CYAN, bold=True)}
    {colorize('>>> !ps aux | grep python', Colors.BRIGHT_BLUE)}
    {colorize('>>> read /var/log/app.log', Colors.BRIGHT_BLUE)}
    {colorize('>>> diff config.old config.new', Colors.BRIGHT_BLUE)}

{colorize('Type any message to start a conversation, or use commands above for file operations.', Colors.BRIGHT_WHITE)}
"""
    print(help_text)

def setup_completion():
    """Setup tab completion for commands and file paths."""
    if not HAS_READLINE:
        return  # Skip completion setup if readline is not available
        
    commands = [
        "read", "write", "append", "diff", "history", "save", "context show", "context clear",
        "context save", "context load", "mode", "model", "exit", "quit", "help", ":r", ":w", ":a", ":x", ":h"
    ]
    def completer(text, state):
        try:
            buf = readline.get_line_buffer()
            if buf.startswith("read ") or buf.startswith("write ") or buf.startswith("append ") or buf.startswith("diff "):
                part = buf.split(" ", 1)[1] if " " in buf else ""
                try:
                    dirname = os.path.dirname(part) or "."
                    prefix = os.path.basename(part)
                    candidates = [os.path.join(dirname, c) for c in os.listdir(dirname) if c.startswith(prefix)]
                except Exception:
                    candidates = []
            else:
                candidates = [c for c in commands if c.startswith(text)]
            try:
                return candidates[state]
            except IndexError:
                return None
        except Exception:
            return None
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")
    if not HAS_READLINE:
        return  # Skip completion setup if readline is not available
        
    commands = [
        "read", "write", "append", "diff", "history", "save", "context show", "context clear",
        "context save", "context load", "mode", "model", "exit", "quit", ":r", ":w", ":a", ":x", ":h"
    ]
    def completer(text, state):
        try:
            buf = readline.get_line_buffer()
            if buf.startswith("read ") or buf.startswith("write ") or buf.startswith("append ") or buf.startswith("diff "):
                part = buf.split(" ", 1)[1] if " " in buf else ""
                try:
                    dirname = os.path.dirname(part) or "."
                    prefix = os.path.basename(part)
                    candidates = [os.path.join(dirname, c) for c in os.listdir(dirname) if c.startswith(prefix)]
                except Exception:
                    candidates = []
            else:
                candidates = [c for c in commands if c.startswith(text)]
            try:
                return candidates[state]
            except IndexError:
                return None
        except Exception:
            return None
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")

def repl(cli: QwenCLI, args: argparse.Namespace):
    """Enhanced REPL with beautiful colorful UI and comprehensive help."""
    setup_completion()
    
    # Show banner on startup
    print_banner()
    
    # Show current configuration with colors
    print(f"{colorize('🔧 Current Configuration:', Colors.BRIGHT_CYAN, bold=True)}")
    print(f"   {colorize('API URL:', Colors.YELLOW)} {colorize(cli.api_url, Colors.WHITE)}")
    print(f"   {colorize('Model:', Colors.YELLOW)} {colorize(cli.model, Colors.BRIGHT_GREEN)}")
    print(f"   {colorize('Profile:', Colors.YELLOW)} {colorize(cli.profile, Colors.BRIGHT_MAGENTA)}")
    print(f"   {colorize('Streaming:', Colors.YELLOW)} {colorize('Enabled' if not args.no_stream else 'Disabled', Colors.BRIGHT_GREEN if not args.no_stream else Colors.BRIGHT_RED)}")
    print(f"   {colorize('Verbose:', Colors.YELLOW)} {colorize('Enabled' if args.verbose else 'Disabled', Colors.BRIGHT_GREEN if args.verbose else Colors.BRIGHT_RED)}")
    print()
    
    print(f"{colorize('💡 Type', Colors.BRIGHT_YELLOW)} {colorize('help', Colors.BRIGHT_CYAN, bold=True)} {colorize('for detailed commands, or start chatting!', Colors.BRIGHT_YELLOW)}")
    print(colorize("=" * 80, Colors.BRIGHT_BLUE))
    
    while True:
        try:
            prompt = f"{colorize('🤖', Colors.BRIGHT_GREEN)} {colorize('>>>', Colors.BRIGHT_CYAN, bold=True)} "
            line = input(prompt).strip()
            if not line:
                continue
            if line in ("exit", "quit"):
                print(f"\n{colorize('💾 Saving session...', Colors.BRIGHT_YELLOW)}")
                cli.save_transcripts()
                cli.save_session()
                print(f"{colorize('👋 Thanks for using QWEN CLI! Goodbye!', Colors.BRIGHT_GREEN, bold=True)}")
                break
            cli.send_user_message(line, stream=not args.no_stream)
        except KeyboardInterrupt:
            print(f"\n{colorize('⚠️  [Interrupted - Press Ctrl+C again to exit, or type', Colors.BRIGHT_YELLOW)} {colorize('exit', Colors.BRIGHT_RED, bold=True)}{colorize(']', Colors.BRIGHT_YELLOW)}")
            continue
        except EOFError:
            print(f"\n{colorize('👋 Goodbye!', Colors.BRIGHT_GREEN)}")
            break
        except Exception as e:
            print(f"{colorize('❌ [Error:', Colors.BRIGHT_RED)} {colorize(str(e), Colors.WHITE)}{colorize(']', Colors.BRIGHT_RED)}")

if __name__ == "__main__":
    # Enable ANSI colors on Windows
    enable_windows_colors()
    
    args = parse_args()

    if args.clear and os.path.exists(SESSION_FILE):
        try:
            os.remove(SESSION_FILE)
            print(colorize("[session cleared]", Colors.BRIGHT_GREEN))
        except Exception:
            pass

    cli = QwenCLI(api_url=args.api_url, model=args.model, profile=args.mode, auto_save=args.auto_save, verbose=args.verbose)
    if args.trust:
        cli.trust_execution = True
    if args.no_stream:
        cli.stream_enabled = False

    if args.verbose:
        print(f"{colorize('[VERBOSE]', Colors.BRIGHT_BLUE)} {colorize('Initialized CLI with API URL:', Colors.WHITE)} {colorize(args.api_url, Colors.BRIGHT_CYAN)}")
        print(f"{colorize('[VERBOSE]', Colors.BRIGHT_BLUE)} {colorize('Model:', Colors.WHITE)} {colorize(args.model, Colors.BRIGHT_GREEN)}")
        print(f"{colorize('[VERBOSE]', Colors.BRIGHT_BLUE)} {colorize('Profile:', Colors.WHITE)} {colorize(args.mode, Colors.BRIGHT_MAGENTA)}")
        print(f"{colorize('[VERBOSE]', Colors.BRIGHT_BLUE)} {colorize('Stream enabled:', Colors.WHITE)} {colorize(str(not args.no_stream), Colors.BRIGHT_GREEN if not args.no_stream else Colors.BRIGHT_RED)}")

    if args.command:
        cli.run_noninteractive(args.command, stream=not args.no_stream)
        sys.exit(0)

    repl(cli, args)
