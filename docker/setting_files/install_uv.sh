#!/bin/bash

# Install Ruff
curl -LsSf https://astral.sh/ruff/$RUFF_VER/install.sh | sh

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setting Python virtual environment
. $HOME/.local/bin/env
uv venv /opt/$VENV_NAME --python $PYTHON_VER

# Set up PATH and activation permanently
echo "export PATH=/opt/$VENV_NAME/bin:\$PATH" >> ~/.zshrc
echo "source /opt/$VENV_NAME/bin/activate" >> ~/.zshrc
export PATH=/opt/$VENV_NAME/bin:$PATH

. /opt/$VENV_NAME/bin/activate

uv pip install  --no-cache-dir -r /tmp/$REQUIREMENTS