"""Train scale 0 with K=32 using the unchanged baseline training loop."""

from config_first_branch import FirstBranchConfig
from train import main


if __name__ == "__main__":
    main(cfg=FirstBranchConfig())
