#! /bin/bash

echo "🚀 Starting setup process..."

# Install uv
echo "📦 Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
echo "✅ uv installed successfully"

# Create and activate virtual environment
echo "🌍 Creating Python virtual environment..."
uv venv .venv
echo "  → Activating virtual environment..."
source .venv/bin/activate
python -m ensurepip --upgrade
echo "✅ Virtual environment created and activated"

git submodule update --init --recursive
uv pip install -e safety-tooling

# Install dependencies
echo "📚 Installing dependencies..."
uv pip install -r requirements.txt
echo "✅ All dependencies installed"

# Setup wandb
echo "🔑 Setting up Weights & Biases..."
echo -n "Would you like to set up Weights & Biases? (y/n): "
read -r SETUP_WANDB
if [[ "$SETUP_WANDB" =~ ^[Yy]$ ]]; then
    echo "Please get your API key from: https://wandb.ai/authorize"
    echo -n "Enter your wandb API key: "
    read -r WANDB_KEY
    wandb login "$WANDB_KEY"
    echo "✅ Successfully logged into wandb"
else
    echo "⏩ Skipping wandb setup"
fi

# Setup Hugging Face
echo "🤗 Setting up Hugging Face..."
echo -n "Would you like to set up Hugging Face? (y/n): "
read -r SETUP_HF
if [[ "$SETUP_HF" =~ ^[Yy]$ ]]; then
    echo "You'll be prompted to enter your Hugging Face token"
    huggingface-cli login
    echo "✅ Hugging Face setup completed"
    echo -n "To also save your Hugging Face token to .env for this project, please re-enter it: "
    read -r HF_TOKEN_FOR_ENV
    if [[ -n "$HF_TOKEN_FOR_ENV" ]]; then
        echo "HF_TOKEN=$HF_TOKEN_FOR_ENV" >> .env
        echo "✅ Hugging Face token written to .env"
    else
        echo "⏩ No token entered, skipping writing Hugging Face token to .env"
    fi
else
    echo "⏩ Skipping Hugging Face setup"
fi

# Setup OpenAI
echo "🤖 Setting up OpenAI..."
echo -n "Would you like to set up OpenAI and save the API key to .env? (y/n): "
read -r SETUP_OPENAI
if [[ "$SETUP_OPENAI" =~ ^[Yy]$ ]]; then
    echo "Please get your OpenAI API key from: https://platform.openai.com/api-keys"
    echo -n "Enter your OpenAI API key: "
    read -r OPENAI_KEY
    if [[ -n "$OPENAI_KEY" ]]; then
        echo "OPENAI_API_KEY=$OPENAI_KEY" >> .env
        echo "✅ OpenAI API key saved to .env"
    else
        echo "⏩ No API key entered, skipping writing OpenAI API key to .env"
    fi
else
    echo "⏩ Skipping OpenAI setup"
fi

echo "🎉 Setup completed successfully!"
echo "💡 Virtual environment is now activated. When opening a new terminal, activate it with: source .venv/bin/activate"