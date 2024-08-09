import argparse
import glob
import json
import tqdm
import os


# Function to read JSON lines from a file
def read_json_lines(file_path, posix_only=False):
    with open(file_path, "r") as file:
        for line in file:
            if "[" not in line:
                if not posix_only or "POSIX" in line:
                    yield json.loads(line.strip())
        # return [json.loads(line) for line in file if line.strip()]


# Main function
def main():
    parser = argparse.ArgumentParser(prog="Trace combining")
    parser.add_argument(
        "-l", "--log_dir", default="./", help="Directory where the trace is stored"
    )
    parser.add_argument(
        "-o", "--output", default="trace.pfw", help="Output combined trace filename"
    )
    parser.add_argument("--posix", action="store_true", help="Include only POSIX calls")
    args = parser.parse_args()

    nf = glob.glob(f"{args.log_dir}/trace-*.pfw")
    if len(nf) == 0:
        raise Exception(
            f"No trace files found in the log_dir specified: {args.log_dir}"
        )

    print(f"Total number of traces: {len(nf)}")

    data = []
    for f in tqdm.tqdm(sorted(nf)):
        data.extend(read_json_lines(f, posix_only=args.posix))

    with open(args.output, "w") as fout:
        # json.dump(data, fout, indent=2)
        json.dump(data, fout)

    print(f"Merging complete. Output file is '{args.output}'.")


if __name__ == "__main__":
    main()
