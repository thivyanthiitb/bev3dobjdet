
# Define local (actually remote in this context) and server paths
LOCAL_USER="umic"
LOCAL_HOST="10.172.4.244"
LOCAL_DIR="/media/umic/my_label/repositories/bevfusion2"
SERVER_DIR="." # The directory on the server where files will be stored

# Define directories and files to sync
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

# Sync directories or files with rsync
for item in "${ITEMS_TO_TRANSFER[@]}"; do
    local_path="$LOCAL_DIR/$item"
    server_path="$SERVER_DIR/${item%/*}/" # Adjust path if necessary

    echo "Syncing $item..."
    rsync -avz --ignore-existing "$LOCAL_USER@$LOCAL_HOST:$local_path" "$server_path"
done

echo "Sync complete."

cp -r ./data/robodrive-sensor/* ./data/nuscenes
echo "Directories arranged."
