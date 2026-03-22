#!/bin/bash
# Score a monitor prompt against labeled sandbagging dataset
# Input: prompt.txt (the monitor prompt template, must contain {task} and {scratchpad})
# Output: F1 score to stdout
#
# Uses Haiku for speed. Scores on 100 balanced samples (50 dec + 50 hon).

PROMPT_FILE="${1:-prompt.txt}"
DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$DIR/eval_prompt.py" --prompt "$PROMPT_FILE" --samples 100 --model haiku
