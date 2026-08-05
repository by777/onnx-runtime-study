# 02_autotvm.py
# Lesson 14: AutoTVM 自动调优 - 实验 2: AutoTVM 自动搜索最优调度
#
# 核心概念:
#   Task        = 一个算子实例（这里: 512x512x512 的 GEMM）
#   搜索空间    = split 的 factor、reorder 顺序、vectorize/parallel 选择 的组合
#   Tuner       = 在搜索空间里采样 + 实测，选出最快的配置
#   XGBTuner    = 用 XGBoost 建模"配置→耗时"，智能搜索（比随机快）
#
# 对比: 手动调度 1.145 ms
# 运行: python 02_autotvm.py  (约 3-5 分钟)
import os

os.environ["TVM_NUM_THREADS"] = "16"  # ← 必须在 import tvm 之前
import time
import numpy as np
import tvm
from tvm import te, autotvm

M, N, K = 512, 512, 512


# -------------1. 定义“可搜索”的GEMM模板 ------------ #
@autotvm.template("gemm_512_512_512")  # 给这个模板起名
def gemm_template():
    # 计算部分：和手动调度完全一样
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)
    # ← 关键：调优器的"答卷"
    cfg = autotvm.get_config()  # 调优器会往cfg里填参数，然后调用模板
    # 注册 3 个可搜索参数（候选值列表）声明"哪些地方可变"
    # 三个参数 = 3 个维度，每个 10 个候选 → 搜索空间 = 10×10×10 = 1000 种组合
    cfg.define_knob("tile_i", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])  # i 分块候选
    cfg.define_knob("tile_j", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])  # j 分块候选
    cfg.define_knob("tile_k", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])  # k 分块候选

    # 可搜索参数1：i轴分块（.val 取调优器选中的值）
    io, ii = s[C].split(C.op.axis[0], cfg["tile_i"].val)

    # 可搜索参数2：j轴分块
    jo, ji = s[C].split(C.op.axis[1], cfg["tile_j"].val)

    # 可搜索参数3： 规约轴k分块
    ko, ki = s[C].split(k, cfg["tile_k"].val)

    # 固定：reorder + vectorize + parallel
    # 标准顺序: io,jo,ko (块外) → ii,ki,ji (块内)
    #   ji 最内层: C 连续写 + B 连续列 load (SIMD)
    #   ki 在 ji 前: A[i][k] 行内连续读 (标量)
    s[C].reorder(io, jo, ko, ii, ki, ji)
    s[C].vectorize(ji)
    s[C].parallel(io)

    return s, [A, B, C]


# -------- 2. 创建Task并调优 -------- #
task = autotvm.task.create("gemm_512_512_512", args=(), target="llvm")
print(f"搜索空间大小: {len(task.config_space)} 种配置")

log_file = "gemm_autotvm.log"
tuner = autotvm.tuner.XGBTuner(task)  # 用 XGBoost 模型做智能搜索

# 开始调优：只测试200个配置
n_trial = 200
print(f"开始调优 {n_trial} 次...")
tuner.tune(
    n_trial=n_trial,  # 用 XGBoost 模型做智能搜索
    measure_option=autotvm.measure_option(
        builder=autotvm.LocalBuilder(),  # 本地编译
        runner=autotvm.LocalRunner(
            number=5, repeat=1, min_repeat_ms=100
        ),  # 实测每个候选的耗时
    ),
    callbacks=[autotvm.callback.log_to_file(log_file)],
)
print("调优完成")

# ------- 3. 使用最优配置编译并测试 -------- #
with autotvm.apply_history_best(log_file):
    with tvm.target.Target("llvm"):
        s, arg_bufs = gemm_template()
        f = tvm.build(s, arg_bufs, target="llvm")


# ---------- 4. 基准测试 ----------
a_np = np.random.rand(M, K).astype("float32")
b_np = np.random.rand(K, N).astype("float32")
a_tvm, b_tvm = tvm.nd.array(a_np), tvm.nd.array(b_np)
c_tvm = tvm.nd.empty((M, N), dtype="float32")

f(a_tvm, b_tvm, c_tvm)  # 预热

t0 = time.time()
for _ in range(100):
    f(a_tvm, b_tvm, c_tvm)
avg_ms = (time.time() - t0) / 100 * 1000

c_np = np.matmul(a_np, b_np)
print(f"AutoTVM 最优 GEMM(512x512x512): {avg_ms:.3f} ms")
print(f"误差: {np.max(np.abs(c_tvm.numpy() - c_np))}")
print(f"\n对比: 手动调度 1.145 ms")
