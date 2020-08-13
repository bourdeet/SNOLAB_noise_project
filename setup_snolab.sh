#!/bin/bash


# Magic line to get the path to the directory containing this script
THIS_DIR="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

echo ""
echo "Setting up 'snolab stuff' {"

# Define location
export SNOLAB_DIR=$THIS_DIR
echo "  SNOLAB_DIR : $SNOLAB_DIR"

# Add this project to the python path
export PYTHONPATH=$PYTHONPATH:$SNOLAB_DIR

echo "}"
