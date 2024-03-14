#!/bin/bash

# Define local (actually remote in this context) and server paths
LOCAL_USER="umic"
LOCAL_HOST="10.172.4.244"
LOCAL_DIR="/media/umic/my_label/repositories/bevfusion2"
SERVER_DIR="." # The directory on the server where files will be stored

# Define directories and files to check and transfer if they don't exist on the server
declare -a ITEMS_TO_TRANSFER=(
    "data/nuscenes/nuscenes_dbinfos_train.pkl"
    "data/nuscenes/nuscenes_gt_database"
    "data/nuscenes/nuscenes_infos_test.pkl"
    "data/nuscenes/nuscenes_infos_train.pkl"
    "data/nuscenes/nuscenes_infos_val.pkl"
    "data/nuscenes/robodrive_infos_test.pkl"
    "data/nuscenes/robodrive-v1.0-test"
    "data/robodrive-sensor"
    "docker_images"
    "pretrained"
)

# Function to pull files or directories
pull_if_not_exists() {
    local item="$1"
    local local_path="$LOCAL_DIR/$item"
    local server_path="$SERVER_DIR/$item"

    # Check if the item is a directory or a file
    if ssh "$LOCAL_USER@$LOCAL_HOST" [ -d "$local_path" ]; then
        # It's a directory, check if it exists on the server
        if [ ! -d "$server_path" ]; then
            # Directory does not exist, pulling it
            echo "Directory $item does not exist on the server. Pulling..."
            scp -r "$LOCAL_USER@$LOCAL_HOST:$local_path" "$SERVER_DIR/${item%/*}/"
        else
            echo "Directory $item already exists on the server. Skipping..."
        fi
    elif ssh "$LOCAL_USER@$LOCAL_HOST" [ -f "$local_path" ]; then
        # It's a file, check if it exists on the server
        if [ ! -f "$server_path" ]; then
            # File does not exist, pulling it
            echo "File $item does not exist on the server. Pulling..."
            scp "$LOCAL_USER@$LOCAL_HOST:$local_path" "$SERVER_DIR/${item%/*}/"
        else
            echo "File $item already exists on the server. Skipping..."
        fi
    else
        echo "$item does not exist on the local machine. Skipping..."
    fi
}

# Loop through each item and transfer it if it does not exist on the server
for item in "${ITEMS_TO_TRANSFER[@]}"; do
    pull_if_not_exists "$item"
done

echo "Transfer complete."
