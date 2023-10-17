#!/usr/bin/env python
import json
import glob
import numpy as np
import tqdm
import os
nf = glob.glob(".trace*.pfw")
data = []
for f in tqdm.tqdm(np.sort(nf)):
    cmd = "sed \"s/}}/}},/g\" %s > %s" %(f, f+".tmp")
    os.system(cmd)
    cmd = "sed -i '$ s/}},/}}/' %s" %(f+".tmp")
    os.system(cmd)
    with open(f+".tmp", "a") as fin:
        fin.write("]")
    with open(f+".tmp", "r") as fin:
        data = data+json.load(fin)


with open("combine.pfw", "w") as fin:
    json.dump(data, fin)
    
