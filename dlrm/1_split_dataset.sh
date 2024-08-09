#!/bin/bash

dataset_dir=$1
split_dir=$2
split_line=${3:-5000000}

if [ ! -d "$dataset_dir" ]; then
    echo "Error: Directory '$dataset_dir' does not exist"
    echo "Usage: $0 <dataset_dir> <split_dir> [split_line]"
    exit 1
fi

if [ ! -f "$dataset_dir/day_0" ]; then
    echo "Error: File 'day_0' does not exist in the directory '$dataset_dir'"
    echo "Usage: $0 <dataset_dir> <split_dir> [split_line]"
    exit 1
fi

mkdir -p "$split_dir"

day_0="$dataset_dir/day_0"
day_0_size=$(stat -c %s "$day_0")
day_0_size_fmt=$(numfmt --to=iec $day_0_size)

echo "Splitting the file 'day_0' ($day_0_size_fmt) into multiple files with $(numfmt --to=si $split_line) lines each"

split -d -l $split_line "$dataset_dir/day_0" "$split_dir/day_0_"

# real    1m48.469s
# user    0m3.135s
# sys     1m44.751s

# Loop through the files and rename or delete them
for file in "$split_dir/day_0_"*; do
    filename=$(basename "$file")
    file_number="${filename#day_0_}"
    file_number=$((10#$file_number))
    if [ "$file_number" -lt 24 ]; then
        new_filename="$split_dir/day_$file_number"
        mv "$file" "$new_filename"
        echo "Renamed '$filename' to '$new_filename'"
    else
        rm "$file"
        echo "Deleted '$filename'"
    fi
done
