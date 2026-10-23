# JP2 导出 PNG 保留输入尺寸

本次修改 `JPEG/export_reconstruction_png.py`，逐张使用解码后 RGB 图片的实际 `(宽, 高)` 保存和校验 PNG。原先代码没有缩放操作，但要求所有输入为 256×256，其他尺寸会触发断言。

## 文件清单与范围

| 文件 | 本次操作 | 用途 |
| --- | --- | --- |
| `export_reconstruction_png.py` | 修改既有未跟踪文件 | 保存实际输入宽高，按该宽高校验输出，导出清单记录实际尺寸，生成说明不再固定写 256×256。 |
| `experiments/reviews/20261022_export_native_size/README.md` | 新增 | 本次使用、验证及审阅记录。 |

开始前脚本已是用户现有的未跟踪文件。`before.json` 保存其修改前全文、SHA-256 及项目 Git 状态；`changes.patch` 仅比较本次操作前后内容，并包含新增本文档的全文。未修改用户已有 JP2、`test_jpeg.py`、旧导出目录或其他既有改动，未暂存、提交或推送。

## 使用

原调用方式继续有效：

```bash
python /workspace/yi/work/JPEG/export_reconstruction_png.py
```

也可指定输入和输出目录：

```bash
python /workspace/yi/work/JPEG/export_reconstruction_png.py \
  --source /workspace/yi/work/JPEG/output_jpeg \
  --output /workspace/yi/work/JPEG/generated_images/custom_export
```

仍沿用 `val_000001.jp2` 至 `val_000024.jp2` 对应 `kodim01.png` 至 `kodim24.png` 的命名。每张图片独立保留宽高，支持同一目录中的横图、竖图、方图及其他尺寸。RGB 解码与无损 PNG 保存流程不变。

## 验证与自动产物

验证包装器使用当前 Python，通过以下命令形式调用导出脚本：

```bash
python -B /workspace/yi/work/JPEG/export_reconstruction_png.py \
  --source <源目录> --output <临时验证目录>
```

执行两个案例，每个案例 24 张图：

1. 当前 `JPEG/output_jpeg`：18 张 768×512、6 张 512×768。
2. 临时混合尺寸 JP2：按 256×256、768×512、512×768、512×716、319×241 循环生成 24 张 RGB 渐变图，JPEG2000 保存参数 `irreversible=False, num_resolutions=3`。

每张导出图校验输入与输出宽高完全一致、RGB 像素逐字节一致、清单中的尺寸与实际一致；生成说明也校验为保留原始宽高。两个案例的测试输入和导出输出均位于自动清理的临时目录。

本目录的 `validation.log` 记录实际执行命令和终端输出；`validation.json` 记录关键生成参数、两次调用的参数、退出码、尺寸分布和每张 PNG 的 SHA-256。它们与 `before.json`、`changes.patch` 均为本次独立审阅记录。

验证结果：两次导出均退出码 0，48 张 PNG 的所有尺寸与像素检查全部通过。补丁通过 `git apply --reverse --check` 核验，不实际应用。
