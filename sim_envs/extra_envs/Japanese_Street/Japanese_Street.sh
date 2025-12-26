#!/bin/sh
UE4_TRUE_SCRIPT_NAME=$(echo \"$0\" | xargs readlink -f)
UE4_PROJECT_ROOT=$(dirname "$UE4_TRUE_SCRIPT_NAME")
chmod +x "$UE4_PROJECT_ROOT/Japanese_Street/Binaries/Linux/Japanese_Street"
"$UE4_PROJECT_ROOT/Japanese_Street/Binaries/Linux/Japanese_Street" Japanese_Street "$@" 
