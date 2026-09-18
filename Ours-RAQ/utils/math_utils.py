import math
import random


def sample_trg(min_k, max_k):
    """Sample a target codebook size as a power of two in [min_k, max_k]."""
    emin, emax = int(math.log2(min_k)), int(math.log2(max_k))
    return 2 ** random.randint(emin, emax)


def powers_of_two(lo: int, hi: int):
    values = []
    value = 1
    while value < lo:
        value <<= 1
    while value <= hi:
        values.append(value)
        value <<= 1
    return values
