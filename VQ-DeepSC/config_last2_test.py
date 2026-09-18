import os

from config import Config, _env_int_list


class Last2CascadeTestConfig(Config):
    """Two-codebook test configuration, independent of the training label."""

    NUM_EMBEDDINGS_LIST = _env_int_list(
        "VQ_DEEPSC_NUM_EMBEDDINGS_LIST", [8, 16]
    )
    EXPERIMENT_NAME = os.environ.get(
        "VQ_DEEPSC_EXPERIMENT_NAME",
        "k" + "_".join(map(str, NUM_EMBEDDINGS_LIST)) + "_last2_copy",
    )
    # Existing training directories retain their historical four-value names.
    CHECKPOINT_DIR = os.environ.get(
        "VQ_DEEPSC_CHECKPOINT_DIR",
        os.path.join(
            "./checkpoints",
            "k64_64_" + "_".join(map(str, NUM_EMBEDDINGS_LIST)) + "_last2_copy",
        ),
    )

    @classmethod
    def validate(cls):
        if len(cls.NUM_EMBEDDINGS_LIST) != 2:
            raise ValueError("Last-two test configuration requires exactly two codebooks")
        for value in cls.NUM_EMBEDDINGS_LIST:
            if value < 2 or value & (value - 1):
                raise ValueError("Every codebook size must be a power of two >= 2")
        if cls.NUM_DOWNSAMPLE_BLOCKS != 4 or len(cls.EMBEDDING_DIM_LIST) != 4:
            raise ValueError("The encoder/decoder backbone must still have four stages")
