"""查表法近似专题 · 公共工具库

所有 step 脚本共享的"建表 + 插值"核心，保证全专题只有一份定义。
对应 T41 回忆里的配置：3bit 段号（8 段）+ 5bit 插值（32 级）。
"""

import math

LOG2E = math.log2(math.e)  # 1.4426950408889634

# ---------------- 设计参数（对应"3bit 查表 + 5bit 插值"） ----------------
SEG_BITS = 3  # 段号位数  → 2^3 = 8 段
FRAC_BITS = 5  # 插值位数  → 2^5 = 32 级
SEGS = 1 << SEG_BITS  # 8
Q = 12  # 表项定点精度：值 = 真值 × 2^Q（Q12）
FRAC_LEVELS = 1 << FRAC_BITS  # 32


def gen_exp2_table(segs=SEGS, q=Q):
    """生成 2^x (x∈[0,1]) 的分段端点表。

    端点 x_k = k/segs，表项 = round(2^{x_k} · 2^q)。
    共 segs+1 项：8 段共享端点，所以是 9 个值不是 16 个。
    """
    return [round((2.0 ** (k / segs)) * (1 << q)) for k in range(segs + 1)]


def exp2_lookup_lerp(f_byte, table, frac_bits=FRAC_BITS):
    """板上运行时：给定 8bit 定点小数 f_byte(= f*256, f∈[0,1))，
    高 3bit 查段、低 5bit 插值，返回 Q12 定点的 2^f 近似值。

    全部整数运算：一次右移取段号、一次掩码取 frac、两次乘法一次加法一次右移。
    两点式：
    y = y0 * (x1 - x) / (x1 - x0) + y1 * (x - x0) / (x1 - x0)
    其中：x1 - x0 = 32 = n
    x - x0 = frac，段内相对偏移
    x1 - x  = n - frac 表示到右端点的距离
    """
    idx = f_byte >> frac_bits  # 高 3bit → 段号 0..7
    frac = f_byte & ((1 << frac_bits) - 1)  # 低 5bit → 0..31
    n = 1 << frac_bits  # 32
    lo, hi = table[idx], table[idx + 1]  # 对应插值公式的y0, y1
    # LERP：权重 (32-frac)/32 与 frac/32，分子相加后 >>5 保持 Q12
    return (lo * (n - frac) + hi * frac) >> frac_bits


#       └ y0 * (x1 - x) /   +  y1 * (x - x0)  ┘  ÷ (x1-x0)
#       └  左端点×到右端距离   +  右端点×到左端距离  ┘  ÷ 段宽


def table_to_c(table, name="exp2_table", suffix="LUT"):
    """把表打印成 C 静态数组（给以后接 C/板子代码用）。"""
    lines = [f"static const int16_t {name}[{len(table)}] = {{"]
    lines.append("    " + ", ".join(f"{v}" for v in table))
    lines.append("};")
    return "\n".join(lines)
