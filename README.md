# Weather Data RAG Explorer

## 🚀 Development Environment Setup

### Prerequisites

- Python Version Management Tool (pyenv recommended)
- Rust compiler
- AWS Account (optional, for full functionality)

### 1. Install Python Version Management

#### macOS (Homebrew)
```bash
# Install pyenv
brew install pyenv

# Add pyenv to shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Reload shell
source ~/.zshrc
```

#### Linux
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Add to shell (adjust for your shell, e.g., .bashrc or .zshrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

### 2. Install Python and Create Virtual Environment

```bash
# Install Python 3.12
pyenv install 3.12.2

# Set local Python version for the project
pyenv local 3.12.2

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Unix/macOS
# or
venv\Scripts\activate  # Windows
```

### 3. Install Project Dependencies

```bash
# Ensure virtual environment is activated
pip install --upgrade pip

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install project dependencies
pip install -r requirements.txt
```

## 🌟 Project Overview

A CLI-based weather data exploration tool using Retrieval-Augmented Generation (RAG) to provide intelligent insights into historical climate data.

## 🖥️ Running the Application

```bash
# Ensure virtual environment is activated
python weather_rag_cli.py

# Specify a different year
python weather_rag_cli.py --year 2020
```

## 🛠️ Development Workflow

```bash
# Activate virtual environment
source venv/bin/activate

# Deactivate when done
deactivate
```

## 🔍 Troubleshooting

### Common Issues
- Ensure correct Python version is used
- Verify virtual environment is activated
- Check Rust installation
- Confirm all dependencies are installed

## 📦 Dependencies
- Python 3.12.x
- Rust
- AWS Bedrock (optional)

## 🤝 Contributing
1. Fork the repository
2. Set up development environment
3. Create feature branch
4. Commit changes
5. Push and create Pull Request

## 📄 License
Distributed under the MIT License. See `LICENSE` for details.