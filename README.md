# AI Agent Playground

A Python project for experimenting with AI agents using different AI Agent framework such as pydantic-ai

## Installation

This project uses [PDM](https://pdm.fming.dev/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-agent-playground.git
cd ai-agent-playground

# Install dependencies
pdm install

# Install with development dependencies
pdm install --dev
```

## Development

### Setup

Make sure you have PDM installed:
```bash
pip install --user pdm
```

### Running Tests

```bash
pdm run pytest
```

### Code Formatting

```bash
# Format code with black
pdm run black .

# Sort imports
pdm run isort .
```
