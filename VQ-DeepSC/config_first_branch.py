"""Scale-0-only K=32 configuration with a 200-epoch training default."""

import os

from config import Config, _env_int_list


class FirstBranchConfig(Config):
    NUM_DOWNSAMPLE_BLOCKS = 1
    EMBEDDING_DIM_LIST = [128]
    NUM_EMBEDDINGS_LIST = _env_int_list("VQ_DEEPSC_NUM_EMBEDDINGS_LIST", [32])
    NUM_EPOCHS = int(os.environ.get("VQ_DEEPSC_NUM_EPOCHS", "200"))
    TEST_DATASET_PATH = os.environ.get(
        "VQ_DEEPSC_TEST_DATASET_PATH", "/workspace/yi/work/Kodak"
    )

    EXPERIMENT_NAME = os.environ.get("VQ_DEEPSC_EXPERIMENT_NAME", "k32_first_branch")
    CHECKPOINT_DIR = os.environ.get(
        "VQ_DEEPSC_CHECKPOINT_DIR", os.path.join("./checkpoints", EXPERIMENT_NAME)
    )
    LOG_DIR = os.environ.get(
        "VQ_DEEPSC_LOG_DIR", os.path.join("./logs", EXPERIMENT_NAME)
    )
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

    @classmethod
    def validate(cls):
        super().validate()
        if cls.NUM_DOWNSAMPLE_BLOCKS != 1 or cls.EMBEDDING_DIM_LIST != [128]:
            raise ValueError("First-branch model requires one scale with 128 channels")
        if cls.NUM_EMBEDDINGS_LIST != [32]:
            raise ValueError("First-branch experiment requires exactly one K=32 codebook")
