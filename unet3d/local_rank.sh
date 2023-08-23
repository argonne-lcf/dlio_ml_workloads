#!/bin/sh
export LOCAL_RANK=$PMI_LOCAL_RANK
echo "LOCAL RANK: $LOCAL_RANK"
$@
