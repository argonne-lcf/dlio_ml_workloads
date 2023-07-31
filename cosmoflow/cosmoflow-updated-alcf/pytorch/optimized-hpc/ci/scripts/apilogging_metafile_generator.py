#!/usr/bin/env python

import itertools
import os

model_name = [i.strip() for i in os.getenv('MODEL_NAME', " ").split(",")]
layouts = [i.strip() for i in os.getenv('APIMETA_LAYOUTS', " ").split(",")]
platforms = [i.strip() for i in os.getenv('APIMETA_PLAYFORMS', " ").split(",")]
precisions = [i.strip() for i in os.getenv('APIMETA_PRECISIONS', " ").split(",")]
ngpu_batch_size = [i.strip().replace('x', ', ') for i in os.getenv('APIMETA_NGPU_BATCH_SIZE', " ").split(",")]

print("model_name, layout, platform, precision, ngpus, batch_size_per_gpu")
for i in itertools.product(model_name, layouts, platforms, precisions, ngpu_batch_size):
    print(", ".join(i))
