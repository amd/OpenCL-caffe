#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver_without_dropout_cpu.prototxt
