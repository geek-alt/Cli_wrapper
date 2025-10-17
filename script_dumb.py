from __future__ import annotations
import os
import sys
import json
import time
import threading
import argparse
import subprocess
import tempfile
import atexit
import itertools
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

# ---------- Basic utilities ----------
HOME = os.path.expanduser("~")
CONFIG_DIR = os.path.join(HOME, ".ai_cli_v5")
SESSION_FILE = os.path.join(CONFIG_DIR, "session.json")
HISTORY_FILE = os.path.join(CONFIG_DIR, "history.txt")
PLUGINS_DIR = os.path.join(CONFIG_DIR, "plugins")
LOG_DIR = os.path.join(CONFIG_DIR, "logs")

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(PLUGINS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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

# ---------- Optional nice UI ----------
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    # Bright colors  
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_WHITE = '\033[97m'
    # Regular colors
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLUE = '\033[34m'
    WHITE = '\033[97m'

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

def colorize(s: str, c: str, bold: bool = False) -> str:
    """Add color to text with optional bold formatting."""
    if not sys.stdout.isatty():
        return s
    formatting = Colors.BOLD if bold else ''
    return f"{formatting}{c}{s}{Colors.RESET}"

# ---------- Optional dependencies auto-install ----------
REQ_PKGS = ["psutil", "torch", "transformers"]
MAYBE_PKGS = ["pygments", "accelerate", "safetensors"]

def ensure_packages(auto_install: bool) -> None:
    """Ensure core packages are available. If not and auto_install True, try to pip install them."""
    missing = []
    for p in REQ_PKGS:
        try:
            __import__(p)
        except Exception:
            missing.append(p)
    if not missing:
        return
    print(colorize("[setup] Missing packages detected: " + ", ".join(missing), Colors.BRIGHT_YELLOW))
    if not auto_install:
        print("[setup] Auto-install disabled (--no-install). Please install packages manually and re-run.")
        sys.exit(1)
    print("[setup] Attempting to install missing packages via pip. This requires network access.")
    for p in missing:
        print(f"[setup] Installing {p} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p])
        except Exception as e:
            print(f"[setup] Failed to install {p}: {e}")
            print("[setup] Please install manually and re-run.")
            sys.exit(1)

# ---------- Importing now that packages may be present ----------
def import_optional(pkgs):
    res = {}
    for p in pkgs:
        try:
            res[p] = __import__(p)
        except Exception:
            res[p] = None
    return res

# ---------- Hardware detection ----------
def detect_hardware() -> Dict[str, Any]:
    import psutil
    hw = {"cpu_logical": psutil.cpu_count(logical=True),
          "cpu_physical": psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True),
          "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
          "swap_gb": round(psutil.swap_memory().total / (1024**3), 1),
          "gpu": False, "gpu_name": None, "gpu_vram_gb": 0.0}
    # torch GPU detection if available
    try:
        import torch
        if torch.cuda.is_available():
            hw["gpu"] = True
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            hw["gpu_name"] = props.name
            hw["gpu_vram_gb"] = round(props.total_memory / (1024**3), 1)
    except Exception:
        # nvidia-smi fallback
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and res.stdout.strip():
                line = res.stdout.strip().splitlines()[0]
                name, vram = [s.strip() for s in line.split(",")]
                hw["gpu"] = True
                hw["gpu_name"] = name
                hw["gpu_vram_gb"] = round(float(vram) / 1024.0, 1)
        except Exception:
            pass
    return hw

# ---------- Model roster & selection ----------
# Each entry: id (HF), name, min_ram_gb, min_vram_gb (0 if not required), notes
MODEL_ROSTER = [
    {"id": "bigcode/smollm-360M", "name": "SmolLM-360M", "min_ram_gb": 4, "min_vram_gb": 0, "notes": "Tiny, CPU-friendly, basic chat/code"},
    {"id": "tiiuae/falcon-7b-instruct", "name": "Falcon-7B-Instruct", "min_ram_gb": 24, "min_vram_gb": 12, "notes": "Strong coding and reasoning if GPU≥12GB"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.2", "name": "Mistral-7B-Instruct", "min_ram_gb": 24, "min_vram_gb": 12, "notes": "Good balance for code & chat"},
    {"id": "meta-llama/Llama-2-7b-chat-hf", "name": "Llama-2-7B", "min_ram_gb": 24, "min_vram_gb": 12, "notes": "Chat & coding, widely used"},
    {"id": "meta-llama/Llama-2-13b-chat-hf", "name": "Llama-2-13B", "min_ram_gb": 64, "min_vram_gb": 24, "notes": "Large, high capability"},
    {"id": "google/gemma-2b-it", "name": "Gemma-2B", "min_ram_gb": 12, "min_vram_gb": 0, "notes": "Small but capable conversational model"},
    {"id": "microsoft/phi-3-mini-1.3B", "name": "Phi-3-mini", "min_ram_gb": 8, "min_vram_gb": 0, "notes": "Efficient small model good for many tasks"},
    {"id": "mistralai/mixtral-8x7b-instruct", "name": "Mixtral-8x7B", "min_ram_gb": 96, "min_vram_gb": 48, "notes": "Top-end, high resources"},
    # Add or reorder models as desired
]

def choose_model_for_hw(hw: Dict[str, Any]) -> Dict[str, Any]:
    candidates = []
    for m in MODEL_ROSTER:
        ok_ram = hw["ram_gb"] >= m["min_ram_gb"]
        ok_vram = (not hw["gpu"]) or (hw["gpu_vram_gb"] >= m["min_vram_gb"])
        # allow CPU-only for models with vram 0 but large RAM
        if ok_ram and ok_vram:
            candidates.append(m)
    if not candidates:
        # fallback: pick smallest by min_ram
        return sorted(MODEL_ROSTER, key=lambda x: x["min_ram_gb"])[0]
    # choose most capable (highest min_ram in roster) that still fits
    return sorted(candidates, key=lambda x: (x["min_ram_gb"], x["min_vram_gb"]))[-1]

# ---------- Universal system prompts (profiles) ----------
SYSTEM_PROFILES = {
    "cli": "You are a concise command-line assistant. Give short, practical, and precise answers. When providing code, show runnable examples.",
    "dev": "You are a developer assistant. Provide detailed code, tests, explanations, and trade-offs. Prioritize correctness and reproducibility.",
    "teacher": "You are a teacher. Explain step-by-step, simplify complex ideas, and provide examples and analogies.",
    "analyst": "You are a data analyst. Focus on data, statistics, reproducible code, and clear interpretation of results.",
}
UNIVERSAL_SYSTEM_PROMPT = (
    "You are a personal assistant capable of: chatting, coding, debugging, solving math and science problems, "
    "explaining history and geography, suggesting recipes, and assisting general everyday tasks. "
    "Adjust responses to the user's level, provide step-by-step when requested, and be concise when asked. "
    "When answering code questions, provide runnable examples and explain edge cases."
)

# ---------- CLI core (conversation, history, commands) ----------
class AICLI:
    def __init__(self, api_backend: str = "transformers", auto_save: bool = True, trust_execution: bool = False, verbose: bool = False):
        self.api_backend = api_backend
        self.auto_save = auto_save
        self.trust_execution = trust_execution
        self.verbose = verbose

        self.model_meta: Optional[Dict[str, Any]] = None
        self.system_prompt = UNIVERSAL_SYSTEM_PROMPT
        self.profile = "cli"

        self.conversation: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.history: List[Dict[str, Any]] = []
        self.max_history = 100

        self.session_file = SESSION_FILE
        self.log_base = os.path.join(LOG_DIR, f"session_{now_ts()}")
        self.plugins = {}
        self._load_session()
        self._load_plugins()

        # Transformer objects set later when model loaded
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def _log(self, *args):
        if self.verbose:
            print("[v]", *args)

    def _start_progress_indicator(self, message: str = "Processing") -> None:
        """Start a progress spinner in a separate thread."""
        if hasattr(self, '_progress_thread') and self._progress_thread and self._progress_thread.is_alive():
            return  # Already running
        
        self._stop_progress = threading.Event()
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
        if hasattr(self, '_stop_progress'):
            self._stop_progress.set()
        if hasattr(self, '_progress_thread') and self._progress_thread and self._progress_thread.is_alive():
            self._progress_thread.join(timeout=0.5)

    # session persistence
    def _load_session(self):
        if os.path.exists(self.session_file):
            try:
                data = json.loads(safe_read(self.session_file))
                if isinstance(data.get("conversation"), list):
                    self.conversation = data["conversation"]
                if isinstance(data.get("history"), list):
                    self.history = data["history"]
                self.profile = data.get("profile", self.profile)
                self.system_prompt = data.get("system_prompt", self.system_prompt)
            except Exception:
                pass

    def save_session(self):
        if not self.auto_save:
            return
        payload = {
            "conversation": self.conversation[-(self.max_history*2):],
            "history": self.history[-self.max_history:],
            "profile": self.profile,
            "system_prompt": self.system_prompt,
            "saved_at": now_ts(),
        }
        try:
            safe_write(self.session_file, json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            pass

    # plugin system (very simple)
    def _load_plugins(self):
        sys.path.insert(0, PLUGINS_DIR)
        for entry in os.listdir(PLUGINS_DIR):
            if not entry.endswith(".py"):
                continue
            name = entry[:-3]
            try:
                mod = __import__(name)
                if hasattr(mod, "register"):
                    try:
                        mod.register(self)
                        self.plugins[name] = mod
                        self._log("plugin loaded:", name)
                    except Exception as e:
                        print(f"[plugin] register failed for {name}: {e}")
                else:
                    self.plugins[name] = mod
            except Exception as e:
                print(f"[plugin] failed to load {name}: {e}")

    # file operations
    def cmd_read(self, path: str):
        if not os.path.exists(path):
            print("[read] file not found")
            return
        try:
            txt = safe_read(path)
            print(txt)
        except Exception as e:
            print("[read error]", e)

    def cmd_write(self, path: str, content: str):
        try:
            safe_write(path, content)
            print(f"[write] {path}")
        except Exception as e:
            print("[write error]", e)

    def cmd_append(self, path: str, content: str):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n" + content)
            print(f"[append] {path}")
        except Exception as e:
            print("[append error]", e)

    def cmd_diff(self, a: str, b: str):
        if not os.path.exists(a) or not os.path.exists(b):
            print("[diff] file not found")
            return
        import difflib
        sa = safe_read(a).splitlines(keepends=True)
        sb = safe_read(b).splitlines(keepends=True)
        diff = difflib.unified_diff(sa, sb, fromfile=a, tofile=b)
        out = "".join(diff)
        if not out:
            print("[diff] no differences")
        else:
            print(out)

    # sandboxed execution
    def execute_code(self, filename_hint: str, code: str, lang: Optional[str] = None):
        if not self.trust_execution:
            if not confirm("Execute code locally? This can run arbitrary code. Continue"):
                print("[execute] cancelled")
                return
        lang = (lang or "").lower()
        if not lang:
            if filename_hint.endswith(".py") or ("def " in code and "import " in code):
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
                res = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=60)
                if res.stdout:
                    print(res.stdout)
                if res.stderr:
                    print("[stderr]", res.stderr)
            except subprocess.TimeoutExpired:
                print("[execute] timeout")
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
                res = subprocess.run(["bash", tmp_path], capture_output=True, text=True, timeout=60)
                if res.stdout:
                    print(res.stdout)
                if res.stderr:
                    print("[stderr]", res.stderr)
            except subprocess.TimeoutExpired:
                print("[execute] timeout")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        else:
            print("[execute] unknown language or not executable")

    # model loading and generation via transformers
    def load_model(self, hf_id: str, trust_install: bool = True):
        """
        Attempt to load model via transformers. Uses device_map='auto' and low_cpu_mem_usage when possible.
        Returns True on success, False otherwise.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except Exception as e:
            print("[load_model] Transformers or torch missing:", e)
            return False

        self._start_progress_indicator(f"Loading tokenizer for {hf_id}")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
            self._stop_progress_indicator()
        except Exception as e:
            self._stop_progress_indicator()
            print(colorize(f"[model] tokenizer load failed: {e}", Colors.BRIGHT_RED))
            return False

        # determine device_map and dtype
        use_cuda = False
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False

        kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
        if use_cuda:
            # prefer float16
            try:
                import torch
                # some models require bf16 vs fp16; we attempt auto dtype detection
                kwargs["torch_dtype"] = torch.float16
            except Exception:
                pass
            device_map = "auto"
        else:
            device_map = None

        self._start_progress_indicator(f"Loading model weights for {hf_id} (this may take several minutes)")
        try:
            model_lm = AutoModelForCausalLM.from_pretrained(hf_id, device_map=device_map, **kwargs)
            pipe = pipeline("text-generation", model=model_lm, tokenizer=tok, device_map=device_map)
            self.tokenizer = tok
            self.model = model_lm
            self.pipeline = pipe
            self.model_meta = {"hf_id": hf_id}
            self._stop_progress_indicator()
            print(colorize(f"[model] ✅ Loaded successfully: {hf_id}", Colors.BRIGHT_GREEN, bold=True))
            return True
        except Exception as e:
            self._stop_progress_indicator()
            print(colorize(f"[model] ❌ Load failed: {e}", Colors.BRIGHT_RED, bold=True))
            return False

    def generate(self, user_message: str, max_tokens: int = 256) -> str:
        # Append to conversation and call pipeline
        self.conversation.append({"role": "user", "content": user_message})
        # For simple usage, craft prompt as system+history+user
        system = self.system_prompt
        prompt = system + "\n\n"
        for msg in self.conversation:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                continue
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += f"Assistant:"

        # If pipeline available:
        if self.pipeline is None:
            return colorize("Model not loaded.", Colors.BRIGHT_RED)
        
        self._start_progress_indicator("Generating AI response")
        try:
            out = self.pipeline(prompt, max_new_tokens=max_tokens, do_sample=False)
            self._stop_progress_indicator()
            text = out[0].get("generated_text", "")
            # try to extract assistant reply after last "Assistant:"
            if "Assistant:" in text:
                reply = text.split("Assistant:")[-1].strip()
            else:
                reply = text.strip()
            self.conversation.append({"role": "assistant", "content": reply})
            self.history.append({"input": user_message, "output": reply, "ts": now_ts()})
            if len(self.history) > self.max_history:
                self.history.pop(0)
            self.save_session()
            return reply
        except Exception as e:
            self._stop_progress_indicator()
            return colorize(f"[generation error] {e}", Colors.BRIGHT_RED)

    # Context management methods
    def context_show(self, n: int = 10) -> None:
        print("--- context (last messages) ---")
        for msg in self.conversation[-n:]:
            print(f"{msg.get('role')}: {msg.get('content')}\n")
        print("--- end ---")

    def context_clear(self) -> None:
        self.conversation = [{"role": "system", "content": self.system_prompt}]
        print("[context cleared]")

    def context_save(self, name: str) -> None:
        path = os.path.join(CONFIG_DIR, f"context_{name}.json")
        safe_write(path, json.dumps(self.conversation, ensure_ascii=False, indent=2))
        print("[context saved]", path)

    def context_load(self, name: str) -> None:
        path = os.path.join(CONFIG_DIR, f"context_{name}.json")
        if not os.path.exists(path):
            print("[context load] not found")
            return
        try:
            self.conversation = json.loads(safe_read(path))
            print("[context loaded]", path)
        except Exception as e:
            print("[context load error]", e)

    def save_transcripts(self) -> None:
        ts = now_ts()
        txt = os.path.join(LOG_DIR, f"transcript_{ts}.log")
        j = os.path.join(LOG_DIR, f"transcript_{ts}.json")
        try:
            with open(txt, "w", encoding="utf-8") as f:
                for h in self.history:
                    f.write(f">>> {h.get('input')}\n{h.get('output')}\n\n")
            with open(j, "w", encoding="utf-8") as f:
                json.dump({"conversation": self.conversation, "history": self.history}, f, ensure_ascii=False, indent=2)
            print("[transcripts saved]", txt, j)
        except Exception as e:
            print("[save transcripts error]", e)

# ---------- High-level flow: auto-detect, select model, permission ----------
def choose_and_confirm_model(auto_install: bool, trust_execution: bool, verbose: bool) -> Dict[str, Any]:
    print(colorize("Detecting hardware...", Colors.BRIGHT_CYAN))
    # ensure psutil/torch present to detect, attempt safe imports if not installed
    try:
        import psutil  # noqa
    except Exception:
        if auto_install:
            print("[setup] Installing missing 'psutil' for hardware detection")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        else:
            print("[setup] psutil missing and auto-install disabled; cannot detect hardware reliably.")
            sys.exit(1)
    try:
        import torch  # noqa
    except Exception:
        if auto_install:
            print("[setup] Installing missing 'torch' for GPU detection/model loading. This may be large.")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        else:
            print("[setup] torch missing and auto-install disabled; GPU detection may not work.")

    hw = detect_hardware()
    print(f"  CPU cores: {hw['cpu_logical']} ({hw['cpu_physical']} physical)")
    print(f"  RAM: {hw['ram_gb']} GB, swap: {hw['swap_gb']} GB")
    if hw["gpu"]:
        print(f"  GPU: {hw['gpu_name']}, VRAM: {hw['gpu_vram_gb']} GB")
    else:
        print("  GPU: none detected")

    best = choose_model_for_hw(hw)
    print()
    print(colorize("Model recommendation:", Colors.BRIGHT_YELLOW), f"{best['name']} ({best['id']})")
    print("Notes:", best.get("notes", ""))
    print()
    print("This operation will:")
    print(" - download the selected model (may be several GB),")
    print(" - write files to disk under ~/.cache/huggingface or transformers default,")
    print(" - use CPU/RAM and GPU if available.")
    print()
    print("Type 'yes' to allow download, disk writes, and model loading. Type anything else to cancel.")
    ans = input(">>> ").strip().lower()
    if ans != "yes":
        print("Permission denied. Exiting.")
        sys.exit(1)
    return best

# ---------- CLI REPL & command handling ----------
def print_banner():
    """Display the colorful AI CLI banner with feature information."""
    banner = f"""
{colorize('╔══════════════════════════════════════════════════════════════════════════════╗', Colors.BRIGHT_CYAN, bold=True)}
{colorize('║', Colors.BRIGHT_CYAN, bold=True)}                          {colorize('🤖 AI CLI v5.0 — Hybrid Assistant', Colors.BRIGHT_YELLOW, bold=True)}                      {colorize('║', Colors.BRIGHT_CYAN, bold=True)}
{colorize('║', Colors.BRIGHT_CYAN, bold=True)}                    {colorize('Local AI Model + Smart Auto-Selection', Colors.BRIGHT_WHITE)}                    {colorize('║', Colors.BRIGHT_CYAN, bold=True)}
{colorize('╚══════════════════════════════════════════════════════════════════════════════╝', Colors.BRIGHT_CYAN, bold=True)}

{colorize('🚀 FEATURES:', Colors.BRIGHT_GREEN, bold=True)}
  {colorize('•', Colors.BRIGHT_GREEN)} {colorize('Hardware Detection & Model Auto-Selection', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_GREEN)} {colorize('Interactive REPL with Conversation Memory', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_GREEN)} {colorize('Code Execution Sandbox (Python/Bash)', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_GREEN)} {colorize('File Operations (Read/Write/Append/Diff)', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_GREEN)} {colorize('Context Management & Session Persistence', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_GREEN)} {colorize('Plugin System Support', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_GREEN)} {colorize('Animated Progress Indicators', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_GREEN)} {colorize('Tab Completion & Command History', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_GREEN)} {colorize('Cross-Platform Color Support', Colors.WHITE)}

{colorize('📋 QUICK COMMANDS:', Colors.BRIGHT_MAGENTA, bold=True)}
  {colorize('read <file>', Colors.CYAN)}       - {colorize('Display file contents', Colors.WHITE)}
  {colorize('write <file> <content>', Colors.CYAN)} - {colorize('Write content to file', Colors.WHITE)}
  {colorize('append <file> <content>', Colors.CYAN)} - {colorize('Append content to file', Colors.WHITE)}  
  {colorize('diff <file1> <file2>', Colors.CYAN)} - {colorize('Show differences between files', Colors.WHITE)}
  {colorize('!<command>', Colors.CYAN)}        - {colorize('Execute shell commands', Colors.WHITE)}
  {colorize('context show/clear/save/load', Colors.CYAN)} - {colorize('Manage conversation context', Colors.WHITE)}
  {colorize('history', Colors.CYAN)}          - {colorize('Show recent conversations', Colors.WHITE)}
  {colorize('save', Colors.CYAN)}             - {colorize('Save session transcripts', Colors.WHITE)}
  {colorize('exit/quit', Colors.CYAN)}        - {colorize('Exit the application', Colors.WHITE)}

{colorize('💡 BEGINNER-FRIENDLY DESIGN:', Colors.BRIGHT_YELLOW, bold=True)}
  {colorize('•', Colors.BRIGHT_YELLOW)} {colorize('No technical setup required - just run and talk!', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_YELLOW)} {colorize('Smart model selection based on your hardware', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_YELLOW)} {colorize('One-time permission for downloads and setup', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_YELLOW)} {colorize('Visual progress indicators for long operations', Colors.WHITE)}
  {colorize('•', Colors.BRIGHT_YELLOW)} {colorize('Type anything to start chatting with AI!', Colors.WHITE)}

"""
    print(banner)

def setup_readline():
    """Setup readline with tab completion and history support."""
    try:
        # Try to import readline - handle different platforms
        rl = None
        try:
            import readline as rl  # type: ignore
        except ImportError:
            # For Windows systems without readline
            try:
                import pyreadline3 as readline  # type: ignore
                rl = readline
            except ImportError:
                print(colorize("[setup] No readline support available - tab completion disabled", Colors.BRIGHT_YELLOW))
                return
        
        if rl is None:
            return
        
        # Load command history with proper error handling
        try:
            if hasattr(rl, 'read_history_file'):
                rl.read_history_file(HISTORY_FILE)  # type: ignore
        except Exception:
            pass
            
        # Register save function with proper error handling
        try:
            if hasattr(rl, 'write_history_file'):
                atexit.register(lambda: rl.write_history_file(HISTORY_FILE))  # type: ignore
        except Exception:
            pass
        
        # Setup tab completion
        def completer(text, state):
            commands = [
                "read", "write", "append", "diff", "context show", "context clear", 
                "context save", "context load", "mode", "model", "history", "save", 
                "exit", "quit", "help", "!", ":r", ":w", ":a", ":x", ":h"
            ]
            
            # Get current buffer to provide context-aware completion
            try:
                buffer = ""
                if hasattr(rl, 'get_line_buffer'):
                    buffer = rl.get_line_buffer()  # type: ignore
                
                # File path completion for file commands
                if buffer.startswith(("read ", "write ", "append ", "diff ")):
                    parts = buffer.split()
                    if len(parts) > 1:
                        path_part = parts[-1]
                        try:
                            import glob
                            dirname = os.path.dirname(path_part) or "."
                            basename = os.path.basename(path_part)
                            pattern = os.path.join(dirname, basename + "*")
                            matches = glob.glob(pattern)
                            # Return relative paths
                            matches = [os.path.relpath(m) for m in matches]
                        except Exception:
                            matches = []
                    else:
                        matches = commands
                else:
                    matches = [cmd for cmd in commands if cmd.startswith(text)]
                
                try:
                    return matches[state]
                except IndexError:
                    return None
            except Exception:
                # Fallback to simple command completion
                matches = [cmd for cmd in commands if cmd.startswith(text)]
                try:
                    return matches[state]
                except IndexError:
                    return None
        
        try:
            if hasattr(rl, 'set_completer') and hasattr(rl, 'parse_and_bind'):
                rl.set_completer(completer)  # type: ignore
                rl.parse_and_bind("tab: complete")  # type: ignore
                print(colorize("[setup] ✅ Tab completion enabled", Colors.BRIGHT_GREEN))
        except Exception:
            pass
        
    except Exception as e:
        print(colorize(f"[setup] Tab completion setup failed: {e}", Colors.BRIGHT_YELLOW))

def repl_loop(cli: AICLI, args):
    setup_readline()
    print_banner()
    print(f"Profile: {cli.profile}  Model: {cli.model_meta.get('hf_id') if cli.model_meta else 'none loaded'}")
    print("Type 'help' for commands.")
    while True:
        try:
            raw = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            print("Exiting. Saving session.")
            cli.save_session()
            cli.save_transcripts()
            break
        if not raw:
            continue
        if raw in ("exit", "quit"):
            print("Goodbye.")
            cli.save_session()
            cli.save_transcripts()
            break
        # simple command parser
        if raw.startswith("read "):
            path = raw.split(" ", 1)[1]
            cli.cmd_read(path); continue
        if raw.startswith("write "):
            parts = raw.split(" ", 2)
            if len(parts) < 3:
                print("Usage: write <path> <content>")
            else:
                cli.cmd_write(parts[1], parts[2])
            continue
        if raw.startswith("append "):
            parts = raw.split(" ", 2)
            if len(parts) < 3:
                print("Usage: append <path> <content>")
            else:
                cli.cmd_append(parts[1], parts[2])
            continue
        if raw.startswith("diff "):
            parts = raw.split(" ", 2)
            if len(parts) < 3:
                print("Usage: diff <file1> <file2>")
            else:
                cli.cmd_diff(parts[1], parts[2])
            continue
        if raw == "history":
            for i, h in enumerate(cli.history[-50:], 1):
                print(f"{i}. [{h.get('ts')}] {h.get('input')}")
            continue
        if raw == "save":
            cli.save_session()
            cli.save_transcripts()
            continue
        if raw.startswith("context "):
            sub = raw.split(" ", 1)[1]
            if sub == "show":
                cli.context_show(10)
            elif sub == "clear":
                cli.context_clear()
            elif sub.startswith("save "):
                name = sub.split(" ",1)[1]
                cli.context_save(name)
            elif sub.startswith("load "):
                name = sub.split(" ",1)[1]
                cli.context_load(name)
            else:
                print("context [show|clear|save <name>|load <name>]")
            continue
        if raw.startswith("mode "):
            mode = raw.split(" ",1)[1]
            if mode in SYSTEM_PROFILES:
                cli.profile = mode
                cli.system_prompt = SYSTEM_PROFILES[mode]
                if cli.conversation and cli.conversation[0].get("role") == "system":
                    cli.conversation[0]["content"] = cli.system_prompt
                else:
                    cli.conversation.insert(0, {"role":"system","content":cli.system_prompt})
                print(f"[mode] set to {mode}")
            else:
                print("Unknown mode. Available:", ", ".join(SYSTEM_PROFILES.keys()))
            continue
        if raw.startswith("model "):
            hf = raw.split(" ",1)[1]
            print("[model] switching to", hf)
            ok = cli.load_model(hf)
            if not ok:
                print("[model] failed to load", hf)
            continue
        if raw.startswith("!"):
            cmd = raw[1:]
            try:
                p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if p.stdout:
                    print(p.stdout.strip())
                if p.stderr:
                    print("[stderr]", p.stderr.strip())
            except Exception as e:
                print("[exec error]", e)
            continue
        if raw == "help":
            print("Commands: read/write/append/diff/context/mode/model/history/save/!<shell>/exit")
            print("Type any other text to send to the model.")
            continue

        # otherwise treat as message to model
        if cli.pipeline is None:
            print("[model] no model loaded. Use 'model <hf_id>' to load or restart CLI to auto-load.")
            continue
        reply = cli.generate(raw)
        print(reply)

# ---------- Methods moved to AICLI class ----------
# Context and transcript methods are now part of the AICLI class

# ---------- Main executable flow ----------
def main():
    # Enable Windows colors first
    enable_windows_colors()
    
    parser = argparse.ArgumentParser(description="AI CLI v5.0 — Hybrid auto-model + developer CLI")
    parser.add_argument("--no-install", dest="no_install", action="store_true", help="Don't auto-install missing Python packages")
    parser.add_argument("--clear", action="store_true", help="Clear saved session on start")
    parser.add_argument("-c", "--command", help="Run single command then exit")
    parser.add_argument("--trust", action="store_true", help="Trust and auto-execute returned code snippets")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Ensure packages
    if not args.no_install:
        ensure_packages(auto_install=True)
    else:
        # minimal check that required pkgs exist
        for p in ("psutil", "torch", "transformers"):
            try:
                __import__(p)
            except Exception:
                print(f"[error] required package {p} missing. Re-run without --no-install or install it manually.")
                sys.exit(1)

    # Detect hardware, choose model, ask permission
    best = choose_and_confirm_model(auto_install=not args.no_install, trust_execution=args.trust, verbose=args.verbose)

    # Instantiate CLI
    cli = AICLI(auto_save=True, trust_execution=args.trust, verbose=args.verbose)

    # Load selected model, try fallbacks if needed
    print(colorize(f"[flow] Loading recommended model {best['name']} ...", Colors.BRIGHT_CYAN))
    ok = cli.load_model(best["id"])
    if not ok:
        print(colorize("[flow] Primary model failed to load. Attempting to find fallback model.", Colors.BRIGHT_YELLOW))
        # try smaller models in roster sorted by min_ram ascending
        sorted_roster = sorted(MODEL_ROSTER, key=lambda x: x["min_ram_gb"])
        fallback_ok = False
        for candidate in sorted_roster:
            if candidate["id"] == best["id"]:
                continue
            print(f"[flow] Trying fallback: {candidate['name']}")
            if cli.load_model(candidate["id"]):
                fallback_ok = True
                break
        if not fallback_ok:
            print(colorize("[flow] All model load attempts failed. You can retry with --no-install off or load a different model with 'model <hf_id>'.", Colors.BRIGHT_RED))
    else:
        print(colorize("[flow] Model loaded and ready.", Colors.BRIGHT_GREEN))

    # Attach metadata for prompt/profile
    cli.profile = "cli"
    cli.system_prompt = SYSTEM_PROFILES.get(cli.profile, UNIVERSAL_SYSTEM_PROMPT)
    if cli.conversation and cli.conversation[0].get("role") == "system":
        cli.conversation[0]["content"] = cli.system_prompt
    else:
        cli.conversation.insert(0, {"role":"system","content":cli.system_prompt})

    # If single command provided, run and exit
    if args.command:
        if cli.pipeline is None:
            print("[flow] model not loaded. Cannot run command.")
            sys.exit(1)
        print("[one-shot] Sending:", args.command)
        out = cli.generate(args.command)
        print(out)
        sys.exit(0)

    # Start REPL
    repl_loop(cli, args)

if __name__ == "__main__":
    main()
