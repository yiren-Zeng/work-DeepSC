# 数学工具的利用集合
import os, math, random, itertools, argparse


def sample_trg(min_k, max_k):
    emin, emax = int(math.log2(min_k)), int(math.log2(max_k))
    return 2 ** random.randint(emin, emax)  # 对数均匀

def powers_of_two(lo: int, hi: int):  # 返回 [lo, hi] 范围内所有 2 的幂次方
    vals = []
    v = 1
    while v < lo:
        v <<= 1
    while v <= hi:
        vals.append(v)
        v <<= 1
    return vals

