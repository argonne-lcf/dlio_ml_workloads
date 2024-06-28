#!/usr/bin/env python

import sys, os, argparse
import pandas as pd
import json
def pfw_to_json(fpfw, fjson):
    with open(fpfw, "r") as fin:
        f = fin.read()
        if f[0]!="[":
            f = "[" + f
        f=f.replace("}}", "}},")
        n=-1
        while f[n]!=",":
            n=n-1
        with open(fjson, "w") as fout:
            fout.write(f[:n]+"]")    
def main():
    parser = argparse.ArgumentParser(prog="pfw_process")
    parser.add_argument("input", help='Input trace file', nargs='+')
    parser.add_argument("--name", '-n', help="The name of the field for process")
    parser.add_argument("--attr", '-a', default="dur", help="The attribute to show")
    parser.add_argument("--operation", '-o', help='operations to perform', default="print")
    parser.add_argument("--skip_head", action='store_true')
    args = parser.parse_args()
    def run(f):
        print(f"Reading tracing information from {f}")
        pfw_to_json(f, "/tmp/tmp.json")
        with open("/tmp/tmp.json", "r") as fin:
            js = json.loads(fin.read())
        qr_df = pd.DataFrame(js)
        if (args.name is not None):
            qr_df = qr_df[qr_df["name"]==args.name]
        if (args.operation == "print"):
            print(qr_df[args.attr].to_string())
        elif (args.operation == "sum"):
            print(f"Sum of {args.attr} with name == {args.name}:", qr_df[args.attr].astype(float).sum())
        elif (args.operation == "average"):
            print(f"Average of {args.attr} with name == {args.name}:", qr_df[args.attr].astype(float).mean())
        else:
            raise Exception("Unknown operations")
        
    for f in args.input:
        try:
            run(f)
        except:
            print(f"error reading {f}")
        
if __name__=="__main__":
    main()
