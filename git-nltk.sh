#!/bin/bash

# Input and output base directories
INPUT_DIR="/tmp/nltk_data/packages"
OUTPUT_DIR="/usr/local/share/nltk_data"

# Create the output directory if it doesn't exist

git clone https://github.com/nltk/nltk_data.git /tmp/nltk_data
mkdir -p "$OUTPUT_DIR"

# Function to process a directory
process_directory() {
    local input_subdir="$1"
    local subdir_name=$(basename "$input_subdir")
    local output_subdir="$OUTPUT_DIR/$subdir_name"
    
    echo "Processing directory: $subdir_name"
    
    # Create corresponding output subdirectory
    mkdir -p "$output_subdir"
    
    # Find all zip files in the input subdirectory and unzip them to the output subdirectory
    find "$input_subdir" -name "*.zip" -type f | while read -r zip_file; do
        echo "  Unzipping: $(basename "$zip_file")"
        unzip -o "$zip_file" -d "$output_subdir"
    done
}

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Find all subdirectories in the input directory and process them
find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r subdir; do
    process_directory "$subdir"
done

echo "All NLTK data has been processed successfully."