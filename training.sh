#!/bin/bash

MYPYTHON="your/python/path"
MYEXE="model_train.py"
MODELPREF="your_model_preference_name"
CHKNAME=${MODELPREF}"_rcut4_nn8.chk.pth.tar"
BESTNAME=${MODELPREF}"_rcut4_nn8.best.pth.tar"
TESTNAME="test_"${MODELPREF}"_rcut4_nn8.csv"
LOGFILE=${MODELPREF}"_rcut4_nn8.results.txt"

CUDA_VISIBLE_DEVICES=0 $MYPYTHON $MYEXE $CHKNAME $BESTNAME $TESTNAME >> $LOGFILE
