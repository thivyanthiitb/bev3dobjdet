#!/bin/bash

# Define source and destination directories
SOURCE_DIR="/path/to/source/folder"
DEST_DIR="/path/to/destination/folder"
LOG_FILE="/path/to/destination/folder/extract.log"

# Create destination directory if it does not exist
mkdir -p "$DEST_DIR"

# Initialize or clear the log file
echo "Starting extraction process..." > "$LOG_FILE"

# Loop through each .tgz file in the source directory
find "$SOURCE_DIR" -name '*.tgz' -print0 | while IFS= read -r -d $'\0' file; do
    echo "Extracting $file..." | tee -a "$LOG_FILE"
    # Use pv to show progress while extracting the .tgz file, and append output to log file
    pv "$file" | tar -xz -C "$DEST_DIR" 2>&1 | tee -a "$LOG_FILE"
    echo "$file extraction completed." | tee -a "$LOG_FILE"
done

echo "All files have been extracted." | tee -a "$LOG_FILE"
