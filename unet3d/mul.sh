#!/bin/bash
MULTIPLY=$1
mkdir -p datax${MULTIPLY}
if [ -z $WORLD_SIZE ]; then
    WORLD_SIZE=1
fi
if [ -z $RANK ]; then
    RANK=0
fi
for n in `seq 0 $((MULTIPLY-1))`
do
    start=$((n*210 + RANK))
    end=$((n*210+209))
    m=0
    for i in $(seq -f "%05g" $start $WORLD_SIZE $end)
    do
	s=$(printf "%05g" $m)
	cp -v data/case_${s}_x.npy datax${MULTIPLY}/case_${i}_x.npy
	cp -v data/case_${s}_y.npy datax${MULTIPLY}/case_${i}_y.npy
	m=$((m+1))
    done
done
	
