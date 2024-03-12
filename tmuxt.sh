#!/bin/bash

SESSION_NAME="thivyanth"

# Check if the tmux session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  # Create a new session if it doesn't exist
  tmux new-session -s $SESSION_NAME -n window0 -d
fi

# Attach to the session
tmux attach -t $SESSION_NAME
