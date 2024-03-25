set -g mouse on


# Unbind the default prefix (Ctrl+B)
unbind C-b

# Set the new prefix to Ctrl+A
set-option -g prefix C-a

# Bind the new prefix so we can send a literal Ctrl+A to applications
bind C-a send-prefix
