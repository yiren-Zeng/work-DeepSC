# RAQ-RVQ 联合 CSCR 四组对照：本次改动审阅记录

日期：2026-10-20。仅修改评估入口与传输标签处理；无需重新训练。

## 使用方式

目标脚本：

```bash
/workspace/yi/work/RAQ-RVQ/scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_combined.sh
```

| CSCR_MODE | 本次运行的唯一方案 | 配置中的 CSCR 状态 |
| --- | --- | --- |
| `off` | 原阶段索引排列，关闭重排 | `OFF` |
| `stage` | 原阶段索引排列，各阶段独立 CSCR | `ON` |
| `joint-pack` | 同尺度联合索引排列，关闭重排 | `OFF` |
| `joint-cscr` | 同尺度联合索引排列，组合码本 CSCR | `ON` |

例如，只运行联合 CSCR：

```bash
CSCR_MODE=joint-cscr bash /workspace/yi/work/RAQ-RVQ/scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_combined.sh
```

也可以在脚本的 `CSCR_MODE="${CSCR_MODE:-}"` 处填写所选模式。
不设置 `CSCR_MODE` 时，原 `USE_CSCR=0/1` 仍分别选择 `off/stage`；显式
`CSCR_MODE` 优先。没有 `all` 模式，不会一次运行四组。
Python 入口新增同名参数 `--cscr-mode`，原 `--cscr/--no-cscr` 继续有效。
原 checkpoint、K 列表、SNR、调制、LDPC、种子、测试集等默认值均保留。
脚本继续只输出终端结果，不保存评估 JSON、图片、码本或 checkpoint 副本。

## 联合表示与计数

- 原 RVQ 仍按原顺序完成残差量化。对同尺度索引对构造 `q=a*K2+b`，不重新寻找
  组合码本中的最近邻。
- 联合码本逐行枚举 `C[a*K2+b]=E1[a]+E2[b]`。对这些**和向量**使用原
  `build_cscr_mapping`：最大 L2 范数起点、余弦贪心排序、Gray 标签。
- LDPC 解码后先逆标签映射，再通过整除和取余拆回原阶段索引，继续调用原重建函数。
- 当前 `[[2,4],[8,2]]` 对应联合 K 为 `[8,16]`。256×256 图像仍为 4096 个信息比特，
  LDPC(128,256)、QPSK 对应 8192 个编码比特、4096 个复符号，实际传输比为 `1/48`。
- 新联合模式要求物理 `stream_packing=combined`；目标脚本原本即使用此模式。
  原有 `per_stage` 方式在旧模式下继续有效。
- 联合模式的传输诊断条目按尺度统计，`per_stage` 中的 `stage=None` 表示一个联合索引，
  不是原来某个阶段。BER 统计实际传输标签比特，索引错误率统计还原后的联合元组。
  原阶段残差与量化统计仍保存在独立的 `rvq_quantization` 字段中。
- helper 支持非空阶段前缀。联合 CSCR 限制每尺度最多 4096 个组合，覆盖当前模型
  两阶段各 K≤64 的范围，并限制余弦矩阵的内存占用；`joint-pack` 不构造组合码本。

## 本次文件清单

以下三项为对会话开始时已有文件的增量修改：

| 文件 | 关键修改 |
| --- | --- |
| [评估入口](../../test_real.py) | 四组模式参数、开关优先级、配置状态和调用转发 |
| [原评估分发](../../evaluation/quality.py) | 新模式转到独立模块，保留原 off/stage 传输路径 |
| [目标 shell 脚本](../../scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_combined.sh) | 新增 `CSCR_MODE` 四选一，兼容 `USE_CSCR` |

以下均为本次新增的未跟踪文件，尚未暂存或提交：

| 文件 | 用途 |
| --- | --- |
| [联合索引工具](../../utils/joint_cscr.py) | 混合进制索引、组合和码本、模式解析、联合 CSCR 映射 |
| [联合评估](../../evaluation/joint_cscr.py) | 联合打包、实际链路、逆标签与原阶段恢复、联合诊断 |
| [联合测试](../../tests/test_joint_cscr.py) | 十项新增测试，包括受损标签对应的物理组合码本重建 |
| [本审阅记录](README.md) | 文件清单、使用方法、实现约定、验证记录 |
| [实际增量补丁](changes.patch) | 相对本次操作前文件内容的差异，包含新增源码和本文档的完整内容 |

修改前已识别工作区已有的修改与未跟踪文件；原 `utils/cscr.py` 和
`tests/test_cscr.py` 当时已存在且未跟踪，本次未修改，不能归入本次新增文件。
补丁相对开始本次任务时的文件内容生成，未把先前 CSCR 实现或其他用户改动归入本次。
`changes.patch` 本身不递归包含其自身。没有执行 `git add/commit/push`。

## 验证结果

执行的单元测试命令：

```bash
cd /workspace/yi/work/RAQ-RVQ
PYTHONDONTWRITEBYTECODE=1 /home/yi/.conda/envs/work/bin/python -B -m unittest discover -s tests -p test_joint_cscr.py -v
PYTHONDONTWRITEBYTECODE=1 /home/yi/.conda/envs/work/bin/python -B -m unittest discover -s tests -v
bash -n scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_combined.sh
```

- 新增 10 项测试全部通过；完整 unittest 发现并运行的 45 项测试全部通过。
- 穷举混合进制往返、单阶段前缀、形状/范围/码本合法性检查通过。
- 四组无错误重建、原残差统计、模型参数和物理长度一致。
- 对人工受损联合标签，实际重建与直接查询物理重排后的组合码本严格一致。
- 实际 shell 启动命令通过捕获执行参数验证：每次只调用一次 `test_real.py`，
  四组选择、旧开关兼容和非法选项拒绝均通过。

GPU 验证在 GPU 0 使用现有真实 checkpoint、`kodim01.png` 和 `kodim02.png`，均为
256×256；模型环境与目标脚本一致，测试数据加载器限定为前两张、workers=0。
命令形式为 `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 ... python -B -`，
以内联 Python 调用真实 `quality.evaluate_no_channel`、`evaluate_ldpc_channel` 和
`test_real.test_real`，并在内存中加载操作前的评估源码做兼容性对照。
关键参数：QPSK、LDPC(128,256)、K=`[[2,4],[8,2]]`、Rician K=1；见下表的信道、
SNR 和种子。结果仅输出终端，无权重、图片、评估日志或 JSON 文件产物；本表保留核查结果。

| 两张图的平均 PSNR / dB | off | stage | joint-pack | joint-cscr |
| --- | ---: | ---: | ---: | ---: |
| AWGN −10 dB，seed 42 | 13.454874 | 13.173361 | 13.276412 | 13.323857 |
| AWGN −10 dB，seed 43 | 13.580995 | 13.361560 | 13.098068 | 13.640204 |
| AWGN 40 dB，seed 42 | 24.182415 | 24.182415 | 24.182415 | 24.182415 |
| Rician −10 dB，seed 42 | 13.295007 | 13.151606 | 12.944430 | 13.524927 |

GPU 核查的关键断言全部通过：

- 原 off/stage 路径与操作前评估函数的完整结果和诊断精确一致。
- 四组无信道重建一致；40 dB 下四组 BER=0，且重建指标等于无信道结果。
- 所有核查条件下四组传输比均为 `1/48`，每图载荷及所有物理长度一致。
- 真实 `test_real` 入口四组选项均生效，配置只出现一次所选模式及正确的 ON/OFF。
- checkpoint 的前后 SHA256 相同：
  `b746a58e9714c5d30ec1a9bb200a5cfb3c4ec9adac510b758c3518e940adad98`。

上述为功能与兼容性验证，不是 24 张图、10 个种子的完整性能报告；小样本低 SNR
结果说明联合 CSCR 并不保证相对原方案始终提升，不能承诺固定的 0.5 dB 增益。
评估收益时分别比较 `joint-cscr` 与 `joint-pack`（重排收益），以及 `joint-pack`
与 `off`（打包变化），再报告相对 `off` 的整体变化。
