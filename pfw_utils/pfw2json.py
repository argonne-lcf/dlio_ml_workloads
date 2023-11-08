#!/usr/bin/env python
import json, os
import argparse
import platform
def pfw_to_json(fpfw, fjson):
    with open(fpfw, "r") as fin:
        f = fin.read()
        if f[0]!="[":
            f = "[" + f
        f = f.replace("}}", "}},")
        n=-1
        while f[n]!=",":
            n=n-1
        with open(fjson, "w") as fout:
            fout.write(f[:n]+"]")
def main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--pfw', "-i", default='.trace-0-of-1.pfw')
    parser.add_argument('--json', "-o", default='trace.json')
    args = parser.parse_args()
    pfw_to_json(args.pfw, args.json)
if __name__=="__main__":
    main()
