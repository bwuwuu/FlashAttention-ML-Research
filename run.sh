#!/bin/bash

# Check if torch is installed
if ! python3 -c "import torch" &> /dev/null; then
    # install torch manually
    echo "PyTorch has not been installed. Install PyTorch first."
else
    echo "PyTorch is already installed."
fi

if ! python3 -c "import ninja" &> /dev/null; then
    pip3 install ninja
    echo "Ninja has been installed."
else
    echo "Ninja is already installed."
fi

python3 test.py

