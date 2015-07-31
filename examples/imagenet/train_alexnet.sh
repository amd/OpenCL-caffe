#!/usr/bin/env sh

GLOG_logtostderr=0 ./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver.prototxt
