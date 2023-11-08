#!/usr/bin/env python
from perfetto.trace_processor import TraceProcessor
import sys, os
def main():
    parser = argparse.ArgumentParser(prog="pfw_process")
    parser.add_argument("--input", '-i', help='Input trace file')
    parser.add_argument("--name", '-n', help="The name of the field for process")
    parser.add_argument("--attr", '-a', default="dur", help="The attribute to show")
    parser.add_argument("--operation", '-o', help='operations to perform', default="print")
    args = parser.parse_args()
    print(f"Reading tracing information from {args.input}")
    tp = TraceProcessor(trace=args.input)
    qr_it = tp.query(f"SELECT name, {args.attr} FROM slice")
    qr_df = qr_it.as_pandas_dataframe()
    qr_df = qr_df[qr_it["name"]==args.name]
    if (args.op == "print"):
        print(qr_df.to_string())
    elif (args.op == "sum"):
        print(qr_df.sum())
    elif (args.op == "average"):
        print(qr_df.sum())
    else:
        raise Exception("Unknown operations")

if __name__=="__main__":
    main()
