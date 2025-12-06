#!/bin/bash

. /opt/$VENV_NAME/bin/activate

uv pip install  --no-cache-dir -r /tmp/$REQUIREMENTS
