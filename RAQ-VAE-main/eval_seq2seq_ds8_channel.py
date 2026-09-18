"""Use the shared channel evaluator with the 1/8 + 1/16 RAQVAE_TWO variant."""
import eval_seq2seq_channel as evaluator
from nets.model_ds8 import RAQVAE_TWO_DS8

evaluator.MODEL_CLASS = RAQVAE_TWO_DS8
evaluator.SOURCE_INDEX_COUNT = 1280
evaluator.OUTPUT_TAG = "seq2seq_ds8"
evaluator.SPATIAL_SCALES = {"bottom": "1/8", "top": "1/16"}


if __name__ == "__main__":
    evaluator.main()

