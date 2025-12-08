#!/bin/bash
export PATH="/root/.local/bin:/usr/local/bin:$PATH"
set -e

. /opt/$VENV_NAME/bin/activate

uv pip install  --no-cache-dir -r /tmp/$REQUIREMENTS
