#!/bin/bash
. ../lesson-05-多输入多输出/.venv/bin/activate
python 00_gen_model.py
# ONNX→MNN
../mnn-src/build/MNNConvert -f ONNX --modelFile dwconv.onnx --MNNModel dwconv.mnn 
# MNN 模型→JSON
../mnn-src/build/MNNDump2Json dwconv.mnn dwconv.json
# 统计算子类型
grep -o '"type": "[^"]*"' dwconv.json | sort | uniq -c
