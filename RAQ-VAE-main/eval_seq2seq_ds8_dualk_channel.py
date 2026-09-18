"""Run the shared channel evaluator for the 1/8 + 1/16 dual-K model."""
import eval_seq2seq_channel as evaluator
from nets.model_ds8_dualk import RAQVAE_TWO_DS8_DUALK

evaluator.MODEL_CLASS = RAQVAE_TWO_DS8_DUALK
evaluator.SOURCE_INDEX_COUNT = 1280
evaluator.BOTTOM_INDEX_COUNT = 1024
evaluator.TOP_INDEX_COUNT = 256
evaluator.OUTPUT_TAG = "seq2seq_ds8_dualk"
evaluator.SPATIAL_SCALES = {"bottom": "1/8", "top": "1/16"}
evaluator.DUAL_K = True


if __name__ == "__main__":
    evaluator.main()

