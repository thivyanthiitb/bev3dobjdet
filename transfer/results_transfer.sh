#!/bin/bash

# Define server directory to sync from
SERVER_DIRECTORY="."

# Define target directory on work machine
TARGET_DIRECTORY="umic@10.172.4.244:/media/umic/my_label/repositories/bevfusion2"

# Use rsync to sync files
# -avz: Verbose, archive mode, compress file data during the transfer
# --ignore-existing: skip updating files that exist on receiver
rsync -avz --ignore-existing "$SERVER_DIRECTORY/runs" "$TARGET_DIRECTORY"
rsync -avz --ignore-existing "$SERVER_DIRECTORY/test" "$TARGET_DIRECTORY"

echo "Sync complete."
