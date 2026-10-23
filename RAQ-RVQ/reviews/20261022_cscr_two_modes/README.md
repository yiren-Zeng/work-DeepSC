# CSCR 双模式清理审阅记录

本次修改以操作开始前的工作区文件为基线；保留已有用户改动，不以 Git HEAD 代替本次基线。未执行 git add、git commit 或 git push。

## 最终控制方式

- Shell 只保留 `CSCR_MODE=off|joint-cscr`。
- Python CLI 只保留 `--cscr-mode off|joint-cscr`；Python 函数参数为 `cscr_mode`。
- 两份 shell 脚本均默认 `joint-cscr`。直接调用 Python CLI/评估函数仍默认 `off`。
- 删除 `USE_CSCR`、`use_cscr`、`--cscr`、`--no-cscr`，以及 `stage`、`joint-pack` 的实现分支。
- `off` 继续使用原始逐级索引；`joint-cscr` 继续使用每尺度的联合索引和余弦贪心/Gray 标签。原有量化、重建、LDPC 和 CBR 计算保持原样。
- uniform 脚本原先默认的 `stage` 模式已删除，其开启默认改为 `joint-cscr`。
- 保留脚本已有的码本、SNR、调制、LDPC、数据集、信道及随机种子配置；未修改训练代码、模型或权重。

## 本次修改的既有文件

| 文件 | 用途/关键改动 | 修改前 Git 状态 |
| --- | --- | --- |
| [mixture 评估脚本](../../scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_combined.sh) | 唯一 CSCR_MODE 控制，仅接收两个模式 | 已跟踪，已有修改 |
| [uniform 评估脚本](../../scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_uniform_channel1_combined.sh) | 统一模式控制和默认值 | 原本未跟踪 |
| [test_real.py](../../test_real.py) | 删除布尔 CLI/函数参数，统一模式透传及 JSON 模式信息；禁用含糊的 CLI 缩写 | 已跟踪，已有修改 |
| [evaluation/quality.py](../../evaluation/quality.py) | 删除逐级 CSCR 分支，保留 off 和 joint-cscr 路由 | 已跟踪，已有修改 |
| [evaluation/joint_cscr.py](../../evaluation/joint_cscr.py) | 联合评估仅执行 joint-cscr，删除仅打包分支 | 原本未跟踪 |
| [utils/joint_cscr.py](../../utils/joint_cscr.py) | 两模式校验，删除布尔开关解析和废弃模式提示 | 原本未跟踪 |
| [utils/cscr.py](../../utils/cscr.py) | 删除逐级码本映射封装，保留联合 CSCR 所需的核心余弦/Gray 映射 | 原本未跟踪 |
| [tests/test_joint_cscr.py](../../tests/test_joint_cscr.py) | 两模式、默认值、损坏比特流重建、物理长度及旧入口拒绝测试；覆盖两份 shell 脚本 | 原本未跟踪 |
| [tests/test_cscr.py](../../tests/test_cscr.py) | 保留核心映射测试，删除已移除模式的专用测试 | 原本未跟踪 |

上述 6 个原本未跟踪文件本次均是修改既有文件，不计作本次新增文件。

## 本次新增的审阅文件

- [changes.patch](changes.patch)：仅包含上述 9 个文件相对于本次开始前内容的实际统一补丁。
- [README.md](README.md)：本审阅记录及完整文件清单。

补丁包含以 Git 工作区根目录为基准的 RAQ-RVQ 路径。已通过以下只读检查，证明补丁与当前修改后文件相符：

```bash
git -C /workspace/yi/work apply --reverse --check RAQ-RVQ/reviews/20261022_cscr_two_modes/changes.patch
```

## 验证结果

1. `python -B -m unittest discover -s tests -v`：43 项通过，19.120 秒。运行环境设置 PYTHONDONTWRITEBYTECODE=1、CUDA_VISIBLE_DEVICES=''、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2、OPENBLAS_NUM_THREADS=1、TF_CPP_MIN_LOG_LEVEL=3。
2. `tests/test_independent_raq_rvq_adaptive_arithmetic.py` 的 10 个独立函数测试：用 unittest.FunctionTestCase 包装执行，9 项通过、1 项既有失败。当前环境未安装 pytest，未安装新依赖。
3. 既有失败 `test_adaptive_channel_defaults_match_dense_combined` 已用修改前 dense 脚本快照复现：adaptive 的 SNRS=8、LDPC_RATE=0.4375，dense 修改前后均为 SNRS=12、LDPC_RATE=0.5；两者 MODULATION 都为 16qam、LDPC_N 都为 256。此次没有调整这些实验参数。
4. 两份 shell 脚本 `bash -n` 通过；7 个 Python 文件的内存编译和行尾空白检查通过。
5. 本次涉及的已跟踪文件 `git diff --check` 通过；真实补丁反向应用检查通过。
6. 全工作区的普通 `git diff --check` 会报告 BPG/JPEG 脚本和既有 BER CSV 的行尾空白；这些文件不是本次改动，未修正或计入本次清单。

## 验证期间观察到的自动缓存更新

最终文件核对还发现以下 4 个原已存在、被 Git 忽略的 Python 字节码缓存发生刷新。它们没有手工改写，作为验证期间观察到的自动生成产物单独记录，不计入上面的 9 个源码/脚本/测试文件，也不包含在源码补丁中。验证命令及参数见上一节；所有代理直接执行的 Python 验证命令均使用了 -B 和 PYTHONDONTWRITEBYTECODE=1，因此不把缓存刷新归因于某一个未经证实的命令。

| 缓存路径 | 字节数（前 → 后） | 核对时 SHA-256 |
| --- | --- | --- |
| [evaluation/__pycache__/joint_cscr.cpython-310.pyc](../../evaluation/__pycache__/joint_cscr.cpython-310.pyc) | 4532 → 4354 | `3132f2289c95209ac9edfe2cc77c056d2215e2f1a98c1e2c364820a8ae820ae3` |
| [evaluation/__pycache__/quality.cpython-310.pyc](../../evaluation/__pycache__/quality.cpython-310.pyc) | 20700 → 19730 | `cb26f0d4f057bbe116ee6ce8528d9aa7a801ca921da32cc45fee209c64e0c97c` |
| [utils/__pycache__/cscr.cpython-310.pyc](../../utils/__pycache__/cscr.cpython-310.pyc) | 3131 → 2040 | `03c72370335e52ddaed7838831a427dae78f9efbe806058c8597139398636248` |
| [utils/__pycache__/joint_cscr.cpython-310.pyc](../../utils/__pycache__/joint_cscr.cpython-310.pyc) | 4425 → 4306 | `1d568a202af9844c4d172afcf2805f60459beeedff56da19e6d05e2a94dbf1fb` |

最终核对的范围为：9 个既有源码/脚本/测试文件修改、4 个既有自动缓存刷新、2 个新增审阅文件；没有删除文件。Git 状态相对于开始时仅新增本目录的 README.md 和 changes.patch 两个未跟踪路径，原有用户状态条目全部保留。

没有写入评估结果、图像或训练产物；测试输出保留在本次会话工具日志中。
