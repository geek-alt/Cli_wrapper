# 🤖 AI CLI Assistant Suite

A comprehensive collection of AI-powered command-line interfaces for different user needs and technical requirements.

## 📦 Available Scripts

### 🌐 **script.py** - Professional HTTP API Client (Qwen CLI v3.0)
**Perfect for: Technical users, LM Studio integration, production environments**

### 🏠 **script_dumb.py** - Self-Contained Local AI (AI CLI v5.0)  
**Perfect for: Beginners, offline usage, no external dependencies**

---

## 🚀 Quick Start

### For Technical Users (HTTP API):
```bash
# Start LM Studio server first, then:
python script.py
```

### For Beginners (Local Models):
```bash
# No setup required - everything is auto-installed:
python script_dumb.py
```

---

## 📋 Feature Comparison Matrix

| Feature | script.py (HTTP API) | script_dumb.py (Local) |
|---------|---------------------|------------------------|
| **🎯 Target Users** | Technical/Advanced | Beginners/Non-technical |
| **🔌 Connection** | HTTP API (LM Studio) | Local model execution |
| **📦 Setup Required** | External server setup | Auto-install everything |
| **💾 Model Storage** | Server-side | Downloaded locally |
| **🌐 Internet Required** | Initial setup only | Model download only |
| **⚡ Performance** | Server-dependent | Hardware-dependent |
| **🎨 UI Features** | Advanced + Streaming | Enhanced + Progress bars |
| **🔧 Customization** | High (plugins/profiles) | Medium (built-in features) |
| **💻 Windows Support** | Full ANSI colors | Full ANSI colors |
| **📊 Progress Indicators** | Animated spinners | Animated spinners |
| **⌨️ Tab Completion** | File path completion | Command completion |
| **📝 Session Management** | Advanced persistence | Basic persistence |
| **🔄 Streaming** | Real-time API streaming | Local generation display |
| **🛡️ Error Recovery** | Robust retry logic | Basic error handling |

---

## 📖 Detailed Documentation

## 🌐 script.py - HTTP API Client (Qwen CLI v3.0)

### 🎯 **Purpose**
Professional-grade CLI for technical users who want to integrate with LM Studio or other OpenAI-compatible API servers.

### ✨ **Key Features**
- 🔗 **HTTP API Integration**: Connects to LM Studio, Ollama, or any OpenAI-compatible server
- 🌊 **Real-time Streaming**: Live token-by-token response streaming
- 🎨 **Advanced UI**: Colorful interface with Windows ANSI support
- 🔄 **Smart Retry Logic**: Exponential backoff with automatic failover
- 🧩 **Plugin System**: Extensible architecture for custom commands
- 📊 **Multiple Profiles**: CLI, Developer, Teacher, and Analyst personalities
- ⌨️ **Tab Completion**: Intelligent file path and command completion
- 📝 **Session Persistence**: Automatic conversation state saving
- 🛠️ **File Operations**: Read, write, diff, and execute code snippets
- 🎪 **Interactive REPL**: Full-featured command-line interface

### 🚀 **Installation & Setup**

#### Prerequisites:
1. **LM Studio** (recommended) or any OpenAI-compatible server
2. **Python 3.8+** with pip

#### Setup:
```bash
# Install optional dependencies for enhanced features
pip install requests pygments

# Start LM Studio and load a model (e.g., Qwen2.5-Coder)
# Default API endpoint: http://localhost:1234/v1/chat/completions
```

#### Quick Start:
```bash
# Interactive mode
python script.py

# Single command
python script.py -c "Explain Python decorators"

# Developer mode with verbose logging
python script.py --mode dev --verbose

# Custom API endpoint
python script.py --api-url http://localhost:11434/v1/chat/completions
```

### 🎛️ **Command Line Options**

```bash
python script.py [OPTIONS]

Core Options:
  --api-url URL         API endpoint (default: localhost:1234)
  --model NAME          Model name (default: qwen/qwen3-coder-30b)  
  --mode PROFILE        Personality: cli/dev/teacher/analyst
  
Behavior:
  -c, --command TEXT    Run single command and exit
  --clear               Clear saved session on startup
  --trust               Auto-execute code without confirmation
  --no-stream          Disable streaming responses
  
Debugging:
  -v, --verbose        Enable detailed logging and progress indicators
  --auto-save          Auto-save session (default: true)
  --no-auto-save      Disable automatic session saving
```

### 🎭 **AI Personalities**

- **`cli`** - Concise, factual responses for command-line usage
- **`dev`** - Detailed code explanations with best practices  
- **`teacher`** - Step-by-step educational explanations
- **`analyst`** - Data-focused responses with statistical reasoning

### 📋 **Built-in Commands**

#### File Operations:
```bash
read <file>                    # Display file with syntax highlighting
write <file> <content>         # Write content to file
append <file> <content>        # Append content to file
diff <file1> <file2>          # Show differences between files
```

#### Conversation Management:
```bash
context show                   # Display recent messages
context clear                  # Clear conversation history
context save <name>            # Save current context
context load <name>            # Load saved context
history                        # Show command history
save                          # Save session transcript
```

#### System Operations:
```bash
!<command>                     # Execute shell commands
mode <profile>                 # Switch AI personality
model <name>                   # Change AI model
help                          # Show detailed help
exit/quit                     # Exit application
```

#### Shortcuts:
```bash
:r → read    :w → write    :a → append
:x → exit    :h → help
```

---

## 🏠 script_dumb.py - Local AI Client (AI CLI v5.0)

### 🎯 **Purpose**
Beginner-friendly, self-contained AI CLI that requires no external server setup. Perfect for users who want AI assistance without technical configuration.

### ✨ **Key Features**
- 🏠 **Fully Local**: No external servers or APIs required
- 🤖 **Auto-Setup**: Automatically installs and configures everything
- 🧠 **Smart Model Selection**: Chooses optimal model for your hardware
- 📊 **Hardware Detection**: Automatically detects GPU/CPU capabilities
- 🎨 **Beautiful UI**: Colorful interface with progress indicators
- ⚡ **Optimized Performance**: Uses GPU acceleration when available
- 💾 **Session Memory**: Remembers conversations between sessions
- 🛠️ **Code Execution**: Run and test code snippets locally
- 📝 **File Management**: Built-in file operations
- 🎪 **Interactive Mode**: Easy-to-use command interface

### 🚀 **Installation & Setup**

#### Prerequisites:
- **Python 3.8+** (that's it!)

#### First Run (Auto-Setup):
```bash
python script_dumb.py
# The script will automatically:
# 1. Install required packages (torch, transformers, etc.)
# 2. Detect your hardware capabilities  
# 3. Recommend and download appropriate AI model
# 4. Start the interactive interface
```

#### Subsequent Runs:
```bash
# Standard interactive mode
python script_dumb.py

# Single command mode
python script_dumb.py -c "Write a Python function to calculate fibonacci"

# Verbose mode (shows what's happening)
python script_dumb.py --verbose

# Skip model auto-loading
python script_dumb.py --no-auto-load
```

### 🎛️ **Command Line Options**

```bash
python script_dumb.py [OPTIONS]

Core Options:
  -c, --command TEXT      Run single command and exit
  --verbose              Show detailed progress and debugging info
  --no-auto-load         Skip automatic model loading on startup
  
Hardware:
  --force-cpu            Force CPU-only mode (disable GPU)
  --model-path PATH      Use specific local model path
  
Behavior:
  --trust                Auto-execute code without confirmation  
  --clear                Clear saved session on startup
  --no-auto-install     Skip automatic package installation
```

### 🧠 **Automatic Model Selection**

The script intelligently chooses models based on your hardware:

#### For NVIDIA GPUs (CUDA):
- **High VRAM (12GB+)**: Qwen2.5-Coder-7B (best quality)
- **Medium VRAM (6-12GB)**: Qwen2.5-Coder-3B (good balance)  
- **Low VRAM (4-6GB)**: Qwen2.5-Coder-1.5B (efficient)

#### For Apple Silicon (MPS):
- **M1/M2/M3 Pro/Max**: Qwen2.5-Coder-7B
- **Base M1/M2/M3**: Qwen2.5-Coder-3B

#### For CPU Only:
- **High RAM (16GB+)**: Qwen2.5-Coder-3B
- **Low RAM (8-16GB)**: Qwen2.5-Coder-1.5B

### 📋 **Built-in Commands**

#### AI Interaction:
```bash
# Just type your question or request:
> Explain how neural networks work
> Write a Python web scraper
> Help me debug this code: [paste code]
```

#### File Operations:  
```bash
read <file>                # View file contents
write <file> <content>     # Create/overwrite file
append <file> <content>    # Add to existing file  
execute <file>            # Run Python/shell script
```

#### Session Management:
```bash
save                      # Save current conversation
load                      # Load previous session
clear                     # Clear conversation history
history                   # View command history
```

#### System Commands:
```bash
!<command>                # Execute shell commands
status                    # Show system and model info
help                      # Display help information  
exit/quit                # Exit application
```

---

## 🔧 Configuration & Customization

### Environment Variables

#### For script.py:
```bash
export QWEN_API_URL="http://localhost:1234/v1/chat/completions"
export QWEN_MODEL="qwen/qwen3-coder-30b"  
export QWEN_MODE="dev"
```

#### For script_dumb.py:
```bash
export AI_CLI_MODEL_PATH="/path/to/local/model"
export AI_CLI_FORCE_CPU="true"
export AI_CLI_VERBOSE="true"
```

### Configuration Directories

#### script.py config: `~/.qwen_cli/`
```
~/.qwen_cli/
├── session.json          # Conversation state  
├── history.txt          # Command history
├── plugins/             # Custom plugins
├── logs/               # Session transcripts
└── context_*.json      # Saved contexts
```

#### script_dumb.py config: `~/.ai_cli_v5/`
```
~/.ai_cli_v5/
├── session.json         # Conversation state
├── history.txt         # Command history  
├── models/             # Downloaded AI models
├── logs/              # Session transcripts
└── cache/            # Tokenizer cache
```

---

## 🎯 Usage Examples

### 🌐 script.py Examples

#### Development Workflow:
```bash
# Start in developer mode
python script.py --mode dev

# In the REPL:
>>> Write a FastAPI endpoint for user authentication
>>> read auth.py
>>> diff auth.py auth_backup.py
>>> context save auth_project
```

#### Single Command Usage:
```bash  
# Quick questions
python script.py -c "Explain Python asyncio"

# Code generation
python script.py -c "Generate a SQLAlchemy model for a blog post"

# File analysis
python script.py -c "Review this code: $(cat myfile.py)"
```

#### Advanced Features:
```bash
# Verbose mode with streaming
python script.py --verbose -c "Complex algorithm explanation"

# Custom model/endpoint
python script.py --api-url http://ollama:11434/v1/chat/completions --model codellama
```

### 🏠 script_dumb.py Examples  

#### First-Time Setup:
```bash
# Initial run (auto-installs everything)
python script_dumb.py
# Follow the prompts to select optimal model for your hardware
```

#### Daily Usage:
```bash
# Quick coding help
python script_dumb.py -c "Create a password generator in Python"

# Interactive problem solving  
python script_dumb.py
>>> I need to scrape data from a website
>>> write scraper.py [generated code]
>>> execute scraper.py
```

#### Advanced Usage:
```bash
# Force specific model
python script_dumb.py --model-path ./my-custom-model

# Verbose debugging
python script_dumb.py --verbose -c "Debug this error: [error message]"

# CPU-only mode
python script_dumb.py --force-cpu
```

---

## 🛠️ Troubleshooting

### Common Issues - script.py

#### Connection Issues:
```bash
# Check if LM Studio is running
curl http://localhost:1234/v1/models

# Try different port
python script.py --api-url http://localhost:11434/v1/chat/completions

# Enable verbose logging
python script.py --verbose -c "test"
```

#### Model Issues:
```bash
# List available models in LM Studio
# Or check the server logs

# Use default model
python script.py --model "default"
```

### Common Issues - script_dumb.py

#### Installation Issues:
```bash
# Manual package installation
pip install torch transformers accelerate psutil

# Force reinstall
python script_dumb.py --no-auto-install
pip install --upgrade torch transformers
```

#### Memory Issues:
```bash
# Use smaller model
python script_dumb.py --force-cpu

# Clear cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.ai_cli_v5/cache/
```

#### Performance Issues:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Use verbose mode to see bottlenecks
python script_dumb.py --verbose
```

---

## 📚 Advanced Features

### Plugin Development (script.py)

Create custom plugins in `~/.qwen_cli/plugins/`:

```python
# ~/.qwen_cli/plugins/myfeature.py
def register(cli):
    def cmd_myfeature(arg):
        print(f"My custom feature: {arg}")
    cli.register_command("myfeature", cmd_myfeature)
```

### Custom Models (script_dumb.py)

Use your own fine-tuned models:

```python
# Place model files in ~/.ai_cli_v5/models/my_model/
python script_dumb.py --model-path ~/.ai_cli_v5/models/my_model
```

---

## 🤝 Contributing

### Bug Reports
1. Include script name (script.py or script_dumb.py)
2. Provide command that caused the issue
3. Include verbose output (`--verbose`)
4. Share system information (OS, Python version, GPU)

### Feature Requests  
1. Specify which script the feature is for
2. Describe the use case and expected behavior
3. Consider if it fits the script's target audience

### Pull Requests
1. Follow existing code style
2. Add appropriate error handling
3. Test on both Windows and Unix systems
4. Update documentation as needed

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Qwen Team** - For the excellent Qwen2.5-Coder models
- **Hugging Face** - For the transformers library and model hosting
- **LM Studio** - For providing an excellent local API server
- **Community** - For feedback and contributions

---

## 📞 Support

### Quick Help
- Run with `--help` flag for basic usage
- Use `--verbose` for debugging information  
- Check the troubleshooting section above

### Documentation  
- Type `help` in the interactive REPL
- Read the built-in command documentation
- Check configuration file comments

### Community
- GitHub Issues for bug reports
- GitHub Discussions for questions and ideas
- Community Discord/forum (if available)

---

*Choose the right tool for your needs: `script.py` for advanced users who want server integration, `script_dumb.py` for beginners who want everything to "just work" locally! 🚀*
