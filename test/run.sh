#!/bin/bash

source common.sh

#go to work directory
cd $MY_SCRATCH

python ${CMSSW_BASE}/src/TTH/GenLevel/test/genLevelAnalysis.py

OUTDIR=$HOME/tth/gc/GenLevel/${TASK_ID}/
mkdir -p $OUTDIR 
OFNAME=$OUTDIR/output_${MY_JOBID}.root
cp Loop/tree.root $OFNAME
