"""Run the channel evaluator for the source-K=16, target-K<=128 dual-K model."""
import eval_seq2seq_channel as evaluator
from nets.model_ds8_dualk_src16_kmax128 import (
    RAQVAE_TWO_DS8_DUALK_SRC16_KMAX128,
)

evaluator.MODEL_CLASS = RAQVAE_TWO_DS8_DUALK_SRC16_KMAX128
evaluator.SOURCE_INDEX_COUNT = 1280
evaluator.BOTTOM_INDEX_COUNT = 1024
evaluator.TOP_INDEX_COUNT = 256
evaluator.OUTPUT_TAG = "seq2seq_ds8_dualk_src16_kmax128"
evaluator.SPATIAL_SCALES = {"bottom": "1/8", "top": "1/16"}
evaluator.DUAL_K = True


if __name__ == "__main__":
    evaluator.main()
