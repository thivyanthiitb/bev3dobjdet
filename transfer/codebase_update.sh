#!/bin/bash
git fetch origin
echo "git fetch origin"

#!/bin/bash

# Assume git fetch origin has been done as needed, before running this script

# Define local (actually remote in this context) and server paths
LOCAL_USER="umic"
LOCAL_HOST="10.172.4.244"
LOCAL_DIR="/media/umic/my_label/repositories/bevfusion2"
SERVER_DIR="." # The directory on the server where files will be stored

# Define directories and files to transfer
declare -a ITEMS_TO_TRANSFER=(
    "docker_images"
    "pretrained"
    "checkpoints"
)

# Function to sync directories or files with rsync
sync_with_rsync() {
    local item="$1"
    local local_path="$LOCAL_DIR/$item"
    local server_path="$SERVER_DIR/$item"

    # Use rsync to sync directories or files
    # -avz: verbose, archive mode, compress file data during the transfer
    # --ignore-existing: skip updating files that exist on receiver
    rsync -avz --ignore-existing "$LOCAL_USER@$LOCAL_HOST:$local_path" "$server_path"
}

# Loop through each item and transfer it using rsync
for item in "${ITEMS_TO_TRANSFER[@]}"; do
    sync_with_rsync "$item"
done

echo "Transfer complete."
