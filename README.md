## llmpeg

LLM-powered FFmpeg command assistant that converts natural language instructions into `ffmpeg` commands.

**Repository:** [https://github.com/imortaltatsu/llmpeg](https://github.com/imortaltatsu/llmpeg)  
**PyPI Package:** [https://pypi.org/project/llmpeg/](https://pypi.org/project/llmpeg/)

### Features
- Conversational interface that converts natural language instructions into `ffmpeg` commands
- GPT-compatible API support (OpenRouter, Ollama, or custom endpoints)
- Interactive file selection when requests are ambiguous
- Automatic command validation and error correction
- Support for compression, conversion, and cropping operations

### Installation

#### From PyPI (Recommended)

**Using pip:**
```bash
pip install llmpeg
```

**Using pipx** (isolated installation, recommended for CLI tools):
```bash
pipx install llmpeg
```

**Using uv:**
```bash
uv pip install llmpeg
```

#### From Git Repository

**Using pip:**
```bash
pip install git+https://github.com/imortaltatsu/llmpeg.git
```

**Using pipx:**
```bash
pipx install git+https://github.com/imortaltatsu/llmpeg.git
```

**Using uv:**
```bash
uv pip install git+https://github.com/imortaltatsu/llmpeg.git
```

**Using uvx (run without installing):**
```bash
uvx git+https://github.com/imortaltatsu/llmpeg.git
```

#### Local Development

```bash
# Clone the repository
git clone https://github.com/imortaltatsu/llmpeg.git
cd llmpeg

# Install in development mode
uv sync
# or
pip install -e .
```

### Setup

After installation, configure your API settings:

```bash
# Interactive setup
llmpeg setup init

# Or use environment variables
export LLMPEG_PROVIDER=openrouter
export LLMPEG_API_KEY=your-api-key
export LLMPEG_MODEL_NAME=gpt-oss:20b
```

### Usage

```bash
# Single command
llmpeg -p "convert sample.png to jpg"

# Interactive mode
llmpeg

# Setup commands
llmpeg setup init    # Configure API settings
llmpeg setup show    # Show current configuration
llmpeg setup test    # Test API connection
```

### Configuration

Configuration is stored in `~/.llmpeg/config.py` or `~/.llmpeg/config.json`.

Supported providers:
- **OpenRouter**: Use `gpt-oss:20b` or other models
- **Ollama**: Local Ollama instance (defaults to `gpt-oss:20b`)
- **Custom**: Any OpenAI-compatible API endpoint

### Requirements

- Python 3.12+
- FFmpeg installed and available in PATH
- API access (OpenRouter API key, Ollama running locally, or custom endpoint)
