#!/bin/bash
PY=~/miniconda3/envs/cuda/bin/python
SP=/home/spencer/Data/correspondence/SPair-71k
run(){ local fam=$1 src=$2 glob=$3 gpu=$4 out=$5
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/diagnose_degeneracy_byfamily.py --family $fam \
    --glob "$glob" --realframes /tmp/probe_indomain/$src --spair $SP --n-pairs 12 \
    2>>/tmp/indomain.err | grep -E "^$fam" >> $out
}
