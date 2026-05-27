import math
import random


def sample_trg(min_k, max_k):
    emin, emax = int(math.log2(min_k)), int(math.log2(max_k))
    return 2 ** random.randint(emin, emax)

def powers_of_two(lo: int, hi: int):
    vals = []
    v = 1
    while v < lo:
        v <<= 1
    while v <= hi:
        vals.append(v)
        v <<= 1
    return vals
