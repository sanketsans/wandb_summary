#!/bin/bash

# Wandb Summary Tool Installation Script

echo "🚀 Installing Wandb Summary Tool..."

# Check if Python 3.8+ is available
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version)"
else
    echo "❌ Python $python_version detected, but $required_version+ is required"
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    
    # Check if Ollama server is running
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "✅ Ollama server is running"
    else
        echo "⚠️  Ollama server is not running"
        echo "   Start it with: ollama serve"
    fi
else
    echo "⚠️  Ollama is not installed"
    echo "   Install from: https://ollama.ai"
    echo "   Then run: ollama pull llama3.2"
fi

# Check if wandb is configured
if python3 -c "import wandb; print('✅ Wandb is available')" 2>/dev/null; then
    echo "✅ Wandb is available"
else
    echo "⚠️  Wandb is not installed or configured"
    echo "   Run: pip install wandb && wandb login"
fi

echo ""
echo "🎉 Installation completed!"
echo ""
echo "📖 Usage examples:"
echo "   python wandb_summary/main.py --entity your_username --project your_project --quick"
echo "   python wandb_summary/example.py"
echo ""
echo "📚 For more information, see README.md" 