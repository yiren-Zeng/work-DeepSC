"""Isolated evaluation entry point; reuses the original two-scale PHY evaluator."""
import sys

sys.dont_write_bytecode = True

import eval_seq2seq_channel as shared


def main(argv=None):
    shared.OUTPUT_TAG = "seq2seq_src256_k4_4096"
    return shared.main(argv, native_resolution=True)


if __name__ == "__main__":
    main()
