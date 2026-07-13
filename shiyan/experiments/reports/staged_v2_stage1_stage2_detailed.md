# staged_v2 Stage1 与 Stage2 详细说明

本文只解释下面两个脚本之间的关系，不涉及 Stage3 和 Stage4。

```bash
scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_stage1_src_teacher_ch64_128.sh
scripts/train/current/run_src64_64_raq2_64_staged_v2_curriculum_stage2_raq_warmup_ch64_128.sh
```

## 一句话理解

Stage1 先训练一个普通的 SimVQ source teacher。

Stage2 再加载 Stage1 的 teacher 权重，冻结 teacher 的 encoder、source codebook、decoder，只训练新增的 RAQ 动态码本生成器，让 RAQ 学会根据 source codebook 生成不同大小的目标码本。

也就是说：

```text
Stage1: 先学一个稳定的 source 表示空间
Stage2: 不改这个 source 表示空间，只训练 RAQ 去模仿/适配它
```

## 关键模块名称

代码里的真实模块名如下：

| 说法 | 代码模块 | 作用 |
| --- | --- | --- |
| Encoder / 图像编码器 | `semantic_encoder` | 把输入图像编码成多层特征 |
| Source quantizer / 源码本量化器 | `vector_quantizers` | 原始 SimVQ 源码本，本方案是两层 `64,64` |
| Decoder / 图像解码器 | `semantic_decoder` | 把量化特征重建成图像 |
| RAQ 模块 | `raqs` | 根据 source codebook 动态生成目标码本 `W_trg` |

不要把 `Source encoder` 和 `Encoder` 理解成两个东西。这里实际上只有一个编码器：`semantic_encoder`。所谓 source path，是指 `semantic_encoder -> vector_quantizers -> semantic_decoder` 这条普通 SimVQ 路径。

## Stage1 训练什么

Stage1 脚本核心配置：

| 配置项 | 值 |
| --- | --- |
| `SIMVQ_EXP_FAMILY` | `shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128` |
| `SIMVQ_TRAIN_BRANCH` | `src` |
| `SIMVQ_USE_RAQ` | `0` |
| `SIMVQ_NUM_EMBEDDINGS_LIST` | `64,64` |
| `SIMVQ_BASE_CHANNELS` | `32` |
| 推导特征维度 | `[64,128]` |
| `SIMVQ_DOWNSAMPLE_STRIDES` | `8,2` |
| `NUM_EPOCHS` | 默认 `200` |
| 预训练 | 不使用，显式 unset |
| 信道课程 | 禁用，start/end 设置为 `1000000/1000001` |

Stage1 是从零训练一个不带 RAQ 的普通 SimVQ 模型。

训练路径：

```text
输入图像 x
  -> semantic_encoder
  -> 两层 source SimVQ 量化器 vector_quantizers，码本大小分别是 64 和 64
  -> semantic_decoder
  -> 重建图像 x_hat_src
```

Stage1 中 `SIMVQ_USE_RAQ=0`，所以模型 forward 到 source 重建后就结束，不会创建 RAQ 分支输出。

训练模块：

| 模块 | 是否训练 |
| --- | --- |
| `semantic_encoder` | 是 |
| `vector_quantizers` 中的 SimVQ 投影层 | 是 |
| `semantic_decoder` | 是 |
| `raqs` | 不存在/不启用 |

注意：SimVQ 的底层 embedding 是冻结的，`ProjectedEmbedding` 中真正训练的是投影层 `proj`。因此源码本不是“整个 embedding 直接训练”，而是“冻结随机底码本 + 可训练投影层”。

## Stage1 损失函数

Stage1 调用的是 `DeepSCLoss.forward(...)` 的普通 source 分支。

总训练损失：

```text
L_stage1 = L_recon_src + L_vq_src
```

其中重建损失：

```text
L_recon_src = MSE(x_hat_src, x)
```

因为当前配置里：

| 损失项 | 权重 |
| --- | --- |
| MSE | `1.0` |
| MS-SSIM | `0.0` |
| LPIPS | `0.0` |

所以 Stage1 的重建损失实际就是 MSE。

每一层 SimVQ 的 VQ loss：

```text
L_vq_i = q_latent_i + commitment_cost * e_latent_i
```

代码里：

```text
q_latent_i = MSE(z_q_i, stopgrad(z_e_i))
e_latent_i = MSE(stopgrad(z_q_i), z_e_i)
commitment_cost = 0.25
```

其中：

| 符号 | 含义 |
| --- | --- |
| `z_e_i` | 第 i 层 encoder 输出特征 |
| `z_q_i` | 第 i 层 source SimVQ 量化后的特征 |
| `stopgrad(...)` | detach，不反传梯度 |

两层 VQ loss 会按课程权重加权：

```text
L_vq_src = w_1(epoch) * L_vq_1 + w_2(epoch) * L_vq_2
```

权重调度：

| 阶段 | epoch 区间，默认 200 epoch | VQ 权重 |
| --- | --- | --- |
| Phase1 | `[0,20)` | `[0.25,0.50]` |
| Phase2 | `[20,80)` | 从 `[0.25,0.50]` 线性退火到 `[0.25,0.25]` |
| Phase3 | `[80,200]` | `[0.25,0.25]` |

Stage1 的 source codebook repulsion 权重是 `0.00`，所以没有源码本排斥损失。

Stage1 的信道概率始终是 0，所以训练的是 clean reconstruction，不走信道扰动。

验证和 best checkpoint：

```text
best_vq_deepsc.pth 的选择依据是 validation reconstruction loss
```

也就是说，保存 best model 时比较的是验证集重建损失 `val_recon`，不是训练总损失，也不是 VQ loss。

Stage1 输出：

```text
checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128_unet2_ds8x2_k64/best_vq_deepsc.pth
```

这个 checkpoint 包含已经训练好的：

```text
semantic_encoder
vector_quantizers
semantic_decoder
```

但它不包含 Stage2 新增 RAQ 模块的有效训练结果，因为 Stage1 不启用 RAQ。

## Stage2 如何接上 Stage1

Stage2 脚本第一行关键依赖：

```bash
STAGE1_CKPT="checkpoints/shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage1_src_teacher_rate044_A_patch_ch64-128_unet2_ds8x2_k64/best_vq_deepsc.pth"
```

然后：

```bash
export SIMVQ_PRETRAINED_CHECKPOINT="${SIMVQ_PRETRAINED_CHECKPOINT:-$STAGE1_CKPT}"
export SIMVQ_ALLOW_PRETRAINED="1"
```

这表示 Stage2 会加载 Stage1 的 best checkpoint。

但是 Stage2 构建的是一个新模型：

```text
同样的 semantic_encoder
同样的 vector_quantizers，仍然是 64,64 source codebook
同样的 semantic_decoder
额外新增 raqs 模块
```

加载 checkpoint 时使用的是“只加载同名且 shape 匹配的参数”：

```text
Stage1 有的同名同形状参数 -> 加载
Stage2 新增的 RAQ 参数 -> Stage1 checkpoint 里没有，所以跳过，保持随机初始化
```

因此 Stage1 和 Stage2 的关系不是“Stage2 从头训练”，也不是“Stage2 继续全模型训练”，而是：

```text
Stage2 = 载入 Stage1 的 source teacher + 新增随机初始化的 RAQ + 只训练 RAQ
```

## Stage2 训练什么

Stage2 脚本核心配置：

| 配置项 | 值 |
| --- | --- |
| `SIMVQ_EXP_FAMILY` | `shiyan_raq_src64-64_raq2-64_staged_v2_curriculum_stage2_raq_warmup_rate044_A_patch_ch64-128` |
| `SIMVQ_TRAIN_BRANCH` | `raq_warmup` |
| `SIMVQ_USE_RAQ` | `1` |
| `SIMVQ_NUM_EMBEDDINGS_LIST` | `64,64` |
| `SIMVQ_RAQ_MIN_TRG` | `2` |
| `SIMVQ_RAQ_MAX_TRG` | `64` |
| `SIMVQ_RAQ_TARGET_LIST` | unset，训练时动态采样 |
| `SIMVQ_RAQ_USE_CURRICULUM` | `1` |
| `SIMVQ_RAQ_RECON_GRAD_MODE` | `dual` |
| `SIMVQ_RAQ_TRAIN_ENCODER` | `0` |
| `SIMVQ_RAQ_LATENT_DISTILL_WEIGHT` | 默认 `1.00` |
| `NUM_EPOCHS` | 默认 `100` |
| 信道课程 | 禁用，start/end 设置为 `1000000/1000001` |

Stage2 的训练参数冻结逻辑来自 `configure_trainable_parameters`。

因为 `SIMVQ_TRAIN_BRANCH="raq_warmup"`：

```text
1. 先把整个模型所有参数 requires_grad=False
2. 再把 model.raqs 的参数 requires_grad=True
3. 不解冻 encoder
4. 不解冻 decoder
5. 不解冻 vector_quantizers
```

Stage2 训练模块：

| 模块 | 是否训练 | 说明 |
| --- | --- | --- |
| `raqs` | 是 | 新增的 RAQ 动态码本生成器 |
| `semantic_encoder` | 否 | 来自 Stage1，冻结，只提供特征 |
| `vector_quantizers` | 否 | 来自 Stage1，冻结，只提供 source codebook 和 source latent |
| `semantic_decoder` | 否 | 来自 Stage1，冻结，用 RAQ 量化特征重建图像 |

这就是 Stage2 叫 warmup 的原因：它只让 RAQ 学会接入已有 teacher，不让 encoder/decoder 一起乱动。

## Stage2 forward 流程

Stage2 每个 batch 内部会同时跑 source path 和 RAQ path。

第一步，source path：

```text
x
  -> semantic_encoder
  -> vector_quantizers 使用 source codebook 量化
  -> 得到：
       z_q_src_list
       source_codebooks_list
       reconstructed_images_src
```

这里的 `reconstructed_images_src` 会被算出来，但 Stage2 的 `raq_warmup` 损失并不会直接优化 source reconstruction。它主要是为了得到 source latent/reference。

第二步，采样 RAQ 目标码本大小：

```text
每一层随机采样一个 K_trg
```

默认 100 epoch 下的 RAQ curriculum：

| epoch 区间 | 每层 K_trg 从哪里采样 |
| --- | --- |
| `[0,10)` | `{32,64}` |
| `[10,40)` | `{8,16,32,64}` |
| `[40,100]` | `{2,4,8,16,32,64}` |

因为模型有两层，所以每次会得到类似：

```text
raq_target_list = [K_trg_layer1, K_trg_layer2]
```

例如：

```text
[64,32]
[8,64]
[2,16]
```

第三步，RAQ 生成目标码本：

```text
source_codebook_i = vector_quantizers[i].transformed_weight()
W_trg_i = raqs[i].generate_codebook_transformer(K_trg_i, source_codebook_i)
```

更直白地说：

```text
Stage1 学好的 64 个 source codeword
  -> 输入 RAQ Transformer
  -> 输出 K_trg 个目标 codeword
```

如果 `K_trg=8`，RAQ 就生成 8 个目标码字；如果 `K_trg=64`，RAQ 就生成 64 个目标码字。

第四步，RAQ path：

```text
encoder_features
  -> 使用 W_trg 做 nearest-neighbor 量化
  -> 得到 z_q_raq_list
  -> semantic_decoder
  -> reconstructed_images_raq
```

Stage2 最终优化的是 RAQ path 的输出。

## Stage2 损失函数

Stage2 使用的是 `DeepSCLoss.forward_raq_only(...)`。

总训练损失：

```text
L_stage2 = L_recon_raq + L_vq_raq + L_distill
```

因为当前配置：

```text
SIMVQ_RAQ_REPULSION_WEIGHT = 0.00
```

所以 RAQ codeword repulsion loss 为 0，不参与实际训练。

### 1. RAQ 重建损失

```text
L_recon_raq = MSE(x_hat_raq, x)
```

其中：

| 符号 | 含义 |
| --- | --- |
| `x` | 原图 |
| `x_hat_raq` | RAQ path 重建图 |

同 Stage1 一样，因为 MS-SSIM 和 LPIPS 权重都是 0，所以重建项实际就是 MSE。

### 2. RAQ VQ loss

每层 RAQ 量化也有 VQ loss：

```text
L_vq_raq_i = q_latent_raq_i + commitment_cost * e_latent_raq_i
```

代码里：

```text
q_latent_raq_i = MSE(z_q_raq_i, stopgrad(z_e_i))
e_latent_raq_i = MSE(stopgrad(z_q_raq_i), z_e_i)
commitment_cost = 0.25
```

两层加权：

```text
L_vq_raq = w_1(epoch) * L_vq_raq_1 + w_2(epoch) * L_vq_raq_2
```

默认 100 epoch 的权重调度：

| 阶段 | epoch 区间 | VQ 权重 |
| --- | --- | --- |
| Phase1 | `[0,10)` | `[0.25,0.50]` |
| Phase2 | `[10,40)` | 从 `[0.25,0.50]` 线性退火到 `[0.25,0.25]` |
| Phase3 | `[40,100]` | `[0.25,0.25]` |

### 3. Latent distillation loss

Stage2 最重要的额外项是 latent distill：

```text
L_distill = lambda_distill * mean_i MSE(z_q_raq_i, stopgrad(z_q_src_i))
```

Stage2 默认：

```text
lambda_distill = 1.00
```

并且 Stage2 脚本 unset 了 final weight 和 decay end，所以这个权重在 Stage2 全程保持 1.00，不衰减。

这个损失的意义是：

```text
RAQ 生成的小/动态目标码本量化出来的 latent
要尽量靠近 Stage1 source teacher 量化出来的 latent
```

也就是让 RAQ 先学会“像 teacher 一样表达图像”，不要一开始就自由漂移。

### 4. Stage2 没有使用的损失项

| 损失项 | Stage2 是否使用 | 原因 |
| --- | --- | --- |
| source reconstruction loss | 否 | `raq_warmup` 走 `forward_raq_only` |
| source VQ loss | 否 | `src_vq_loss` 在 `forward_raq_only` 中置 0 |
| RAQ repulsion loss | 否 | `SIMVQ_RAQ_REPULSION_WEIGHT=0.00` |
| source codebook repulsion | 否 | `SIMVQ_SRC_CODEBOOK_REPULSION_WEIGHT=0.00` |
| joint-lite anchor loss | 否 | Stage2 不是 `raq_jointlite` 分支 |

## Stage1 和 Stage2 的真实连接关系

可以把两阶段理解成这样：

```text
Stage1 训练完成：

semantic_encoder  已训练
vector_quantizers 已训练，source codebook = 64,64
semantic_decoder  已训练
raqs              无

        |
        | best_vq_deepsc.pth
        v

Stage2 初始化：

semantic_encoder  从 Stage1 加载，然后冻结
vector_quantizers 从 Stage1 加载，然后冻结
semantic_decoder  从 Stage1 加载，然后冻结
raqs              新建，随机初始化，然后训练
```

Stage2 不是改变 Stage1 teacher，而是在 teacher 旁边接一个 RAQ 生成器：

```text
teacher source codebook C_src
        |
        v
RAQ generator 生成 W_trg(K)
        |
        v
用 W_trg 量化 encoder 特征
        |
        v
冻结的 decoder 重建图像
```

Stage2 的目标是让这个新 RAQ 生成器产生的目标码本 `W_trg` 足够好，使得：

```text
1. RAQ 重建图像 x_hat_raq 接近原图 x
2. RAQ latent z_q_raq 接近 Stage1 teacher latent z_q_src
```

## 为什么 Stage2 要冻结 encoder/decoder

如果 Stage2 一开始就同时训练 RAQ、encoder、decoder，会有一个问题：

```text
RAQ 还没学会生成稳定目标码本，
encoder/decoder 又在变化，
训练目标会同时移动，容易不稳定。
```

所以 Stage2 的设计是：

```text
固定 Stage1 已经学好的表示空间和 decoder
只让 RAQ 学会在这个固定空间里生成可用目标码本
```

这就是 warmup 的意思。

## Stage2 为什么需要 source path

Stage2 虽然不训练 source path，但 forward 里仍然要跑 source path，原因有两个：

1. 需要 `source_codebook_i` 来生成 RAQ 目标码本 `W_trg_i`。
2. 需要 `z_q_src_i` 作为 latent distill 的 teacher target。

所以 Stage2 里的 source path 是“教师/参考路径”，不是“被优化路径”。

## Stage1 与 Stage2 的配置对比

| 项 | Stage1 | Stage2 |
| --- | --- | --- |
| `SIMVQ_TRAIN_BRANCH` | `src` | `raq_warmup` |
| `SIMVQ_USE_RAQ` | `0` | `1` |
| 是否加载预训练 | 否 | 是，加载 Stage1 best |
| 训练模块 | encoder + source quantizer + decoder | 只训练 RAQ |
| 冻结模块 | 无特殊冻结 | encoder/source quantizer/decoder |
| 主要重建目标 | source reconstruction | RAQ reconstruction |
| VQ loss | source VQ | RAQ VQ |
| latent distill | 无 | 有，权重 1.00 |
| RAQ K | 无 | 每层动态采样，范围 2 到 64 |
| 信道扰动 | 关闭 | 关闭 |
| 默认 epochs | 200 | 100 |
| 学习率 | `5e-5` / proj `2e-4` | `5e-5` / proj `2e-4` |

## 最小伪代码

Stage1：

```python
model = DeepSC(use_raq=False)

for x in train_loader:
    out = model.forward_train(x)
    x_hat = out["reconstructed_images"]
    vq_losses = out["vq_losses"]

    loss_recon = mse(x_hat, x)
    loss_vq = weighted_sum(vq_losses)
    loss = loss_recon + loss_vq
    loss.backward()
```

Stage2：

```python
model = DeepSC(use_raq=True)
load_matching_weights(model, stage1_best_checkpoint)

freeze(model)
unfreeze(model.raqs)

for x in train_loader:
    K_list = sample_raq_target_list_by_curriculum(epoch)
    out = model.forward_train(x, raq_trg_list=K_list)

    x_hat_raq = out["reconstructed_images_raq"]
    vq_losses_raq = out["vq_losses_raq"]
    z_q_src = out["z_q_src_list"]
    z_q_raq = out["z_q_raq_list"]

    loss_recon = mse(x_hat_raq, x)
    loss_vq = weighted_sum(vq_losses_raq)
    loss_distill = mse(z_q_raq, stopgrad(z_q_src))
    loss = loss_recon + loss_vq + loss_distill
    loss.backward()
```

## 最核心结论

Stage1 是“老师”的训练：学好一个 `64,64` 的 source SimVQ 表示空间。

Stage2 是“学生模块”的 warmup：加载老师，冻结老师，只训练 RAQ，让 RAQ 生成的动态码本在老师的表示空间里可用。

Stage2 的 RAQ 不是替换 Stage1，而是依赖 Stage1：

```text
没有 Stage1 的 source codebook，RAQ 不知道从什么 codebook 生成目标码本；
没有 Stage1 的 source latent，latent distill 没有 teacher target；
没有 Stage1 的 decoder，RAQ 生成的 latent 无法被稳定地映射回图像。
```
