#!/bin/bash

# The MIT License (MIT)
#
# Copyright (c) 2020-2022 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

# weak scaling copy
for file in $(ls ${OUTPUT_DIR}/logs/*instance*.log); do
    instance_id=$(echo ${file} | awk '{split($1,a,"instance"); split(a[2], b, ".log"); print b[1]}')
    target="${LOGFILE_BASE}_${instance_id}_${EXP_ID}.log"
    if [ ! -f ${target} ]; then
	cp ${file} ${target}
    fi
done

# strong scaling copy
if [ -f ${OUTPUT_DIR}/logs/${RUN_TAG}_${EXP_ID}.log ]; then
    target="${LOGFILE_BASE}_${EXP_ID}.log"
    if [ ! -f ${target} ]; then
	cp ${OUTPUT_DIR}/logs/${RUN_TAG}_${EXP_ID}.log ${target}
    fi
fi
