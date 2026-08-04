# 02_from_onnx.py
# Lesson 13: AI 编译器入门（TVM） - 实验2：整图编译 ONNX → Relay → 执行
#
# 对应编译器架构：前端（把 ONNX 翻译成 Relay IR）
# 之前实验1是"手写算子"（算子级编译）
# 这次是"整图编译"（从 ONNX 模型文件到可执行模块）
#
# 运行： python 02_from_onnx.py

import numpy as np
import onnx
import tvm
from tvm import relay
from tvm.contrib import graph_executor
