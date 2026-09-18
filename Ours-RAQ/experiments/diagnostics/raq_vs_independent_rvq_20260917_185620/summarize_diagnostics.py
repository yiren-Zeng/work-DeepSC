from pathlib import Path
import json,hashlib,csv,subprocess
OUT=Path(__file__).resolve().parent
ROOT=Path('/workspace/yi/work/shiyan')
raq=json.loads((OUT/'raq_diagnostics.json').read_text())
rvq=json.loads((OUT/'independent_rvq_diagnostics.json').read_text())
rvq['drop_scale1_stage2_probe']=json.loads((OUT/'stage_drop_probe.json').read_text())
rvq['results']=[r for r in rvq['results'] if r['label']!='old_layout_channel']
rvq['results'].append(json.loads((OUT/'old_layout_channel.json').read_text()))
(OUT/'independent_rvq_diagnostics.json').write_text(json.dumps(rvq,ensure_ascii=False,indent=2))
def get(report,label):
 return next(r for r in report['results'] if r['label']==label)
def hashfile(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
before=json.loads((OUT/'original_tree_before.json').read_text())
after={}
for p in ROOT.rglob('*'):
 st=p.lstat();after[str(p.relative_to(ROOT))]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'mode':st.st_mode}
changed=[key for key in set(before)|set(after) if before.get(key)!=after.get(key)]
hashes=json.loads((OUT/'source_sha256_before.json').read_text())
hash_changed=[path for path,h in hashes.items() if hashfile(path)!=h]
verification={'original_tree_entries_before':len(before),'original_tree_entries_after':len(after),
 'original_tree_changed':changed,'selected_original_code_and_checkpoints_hash_changed':hash_changed,
 'all_original_files_unchanged':not changed and not hash_changed,
 'training_gpu2_process_running':subprocess.run(['kill','-0','1279601'],capture_output=True).returncode==0,
 'evaluation_worker_exit_codes':json.loads((OUT/'completion.json').read_text()),
 'dataset_images_each_model':[len(raq['image_files']),len(rvq['image_files'])],
 'same_dataset_image_order':raq['image_files']==rvq['image_files'],
 'stage_drop_probe_grad_disabled':rvq['drop_scale1_stage2_probe']['grad_disabled']}
(OUT/'verification.json').write_text(json.dumps(verification,indent=2))
assert verification['all_original_files_unchanged'],verification
assert all(v==0 for v in verification['evaluation_worker_exit_codes'].values())
assert verification['same_dataset_image_order']
q0=get(raq,'default_channel');r0=get(rvq,'default_channel')
qn=get(raq,'default_no_channel');rn=get(rvq,'default_no_channel')
oldn=get(rvq,'old_layout_no_channel');oldc=get(rvq,'old_layout_channel')
ql=get(raq,'matched_low_channel');rl=r0
qnl=get(raq,'matched_low_no_channel')
diff={'default_channel_raq_minus_rvq_psnr':q0['psnr']-r0['psnr'],
 'default_no_channel_raq_minus_rvq_psnr':qn['psnr']-rn['psnr'],
 'matched_low_channel_raq_minus_rvq_psnr':ql['psnr']-rl['psnr'],
 'matched_low_no_channel_raq_minus_rvq_psnr':qnl['psnr']-rn['psnr'],
 'same_rvq_weights_old_minus_current_no_channel_psnr':oldn['psnr']-rn['psnr'],
 'default_channel_psnr_loss_raq':qn['psnr']-q0['psnr'],
 'default_channel_psnr_loss_rvq':rn['psnr']-r0['psnr'],
 'matched_high_no_channel_differences':{}}
for label in ['matched_high_8x4_no_channel','matched_high_4x8_no_channel','matched_high_16x2_no_channel']:
 diff['matched_high_no_channel_differences'][label]=qn['psnr']-get(rvq,label)['psnr']
(OUT/'summary.json').write_text(json.dumps(diff,indent=2))
lines=[
'检查结果：当前 RVQ 的真实码本配置与旧结果不一致，且两个脚本的默认 SNR 与 LDPC 码率不同；在已测的等信息量条件下，普通 RAQ 依然优于这些 RVQ 配置。',
'',
'本次使用原始 shiyan 中两份 best 权重，在 Kodak-256-transform-resize 的全部 24 张 256×256 RGB 图片上复测，种子 42。GPU 0 和 GPU 1 均为 NVIDIA L40S，仅用于评测；GPU 2 的 res6 训练继续运行。shiyan 原目录的全部条目大小、mtime、权限及所核查代码/权重 SHA256 均未改变。',
'',
'**当前脚本的实际默认值**',
'',
'| 项目 | 普通 RAQ | 独立 RAQ-RVQ |',
'|---|---|---|',
'| 实际目标码本 | [32,16] | [[4,2],[8,2]] |',
'| 源码本 | [64,64] | [64,64] |',
'| 编码器/解码器残差块 | 4/4 | 4/4 |',
'| 两尺度特征维度 | 256,512 | 256,512 |',
'| SNR | 6 dB | 3 dB |',
'| LDPC n,k,rate | 256,192,0.75 | 256,128,0.5 |',
'| 调制 | QPSK | QPSK |',
'| 信息 bit/图 | 6144 | 4096 |',
'| 源编码 bpp | 0.09375 | 0.0625 |',
'| 编码后发送 bit/图 | 8192 | 8192 |',
'| QPSK 符号/图 | 4096 | 4096 |',
'| CBR（符号/RGB元素） | 1/48 | 1/48 |',
f"| 本次默认 PSNR | {q0['psnr']:.6f} | {r0['psnr']:.6f} |",
f"| 本次默认 MS-SSIM | {q0['ms_ssim']:.6f} | {r0['ms_ssim']:.6f} |",
'',
'普通 RAQ 的平坦索引也实际合并为一条 LDPC 流；脚本/JSON 的 per_stage 标签在这个非 RVQ 分支上不意味着分开编码。因此两者默认发送 bit 和 CBR 完全相同，但普通 RAQ 利用更高编码率承载了 50% 更多的信息 bit，且 SNR 高 3 dB。不能把这种配置差异当作仅增加 RVQ 的消融。',
'',
'RVQ 脚本中的 RAQ_TARGET_LIST=[16,4] 只是名义元数据，实际独立 RVQ 的四级大小由 INDEPENDENT_RAQ_RVQ_K_LISTS 决定；不能用这个平坦列表计算 RVQ 的发送 bit。另核查了 Ours-RAQ 中同名的 4 块脚本，那里当前默认是 [64,64]、SNR10、16QAM、LDPC0.5，和本次指定的 shiyan 脚本也不同。如果混用了副本结果，会进一步改变比较条件；本次表格按用户指定的两份 shiyan 脚本统计。',
'',
f"关闭信道时，默认普通 RAQ 为 {qn['psnr']:.6f} dB，默认 RVQ 为 {rn['psnr']:.6f} dB。各自打开默认链路仅损失 {diff['default_channel_psnr_loss_raq']:.6f} / {diff['default_channel_psnr_loss_rvq']:.6f} dB。当前约 1.68 dB 的差距主要在源编码/量化端，不能主要归因于 SNR 差异。",
'',
'**同一 RVQ 权重恢复旧配置**',
'',
'RVQ 脚本第 2 行注释和默认文件名仍写 4×64、LDPC 3/4；但第 32 行实际默认 4,2;8,2，第 77 行实际传入 0.5。历史 per_stage JSON 保留了真正的旧配置 [[4,64],[8,2]]、LDPC 0.75。在 SNR 6、9 时分别为 25.559039 / 25.562955 dB。当前 combined JSON 则记录 [[4,2],[8,2]]、LDPC 0.5、SNR 3、23.236794 dB。',
'',
'| RVQ 设置（同一份权重） | 信息 bit/图 | 无信道 PSNR | 无信道 MS-SSIM |',
'|---|---:|---:|---:|',
f"| 当前 [[4,2],[8,2]] | 4096 | {rn['psnr']:.6f} | {rn['ms_ssim']:.6f} |",
f"| 旧 [[4,64],[8,2]] | 9216 | {oldn['psnr']:.6f} | {oldn['ms_ssim']:.6f} |",
'',
f"只恢复首尺度的第二级码本 2→64，每个首尺度索引增加 5 bit，共增加 5120 bit/图；信息量成为原来的 2.25 倍，无信道 PSNR 提升 {diff['same_rvq_weights_old_minus_current_no_channel_psnr']:.6f} dB。恢复旧设置并使用当前 combined 链路、SNR 6、LDPC 0.75，复测为 {oldc['psnr']:.6f} dB、MS-SSIM {oldc['ms_ssim']:.6f}，CBR=1/32。这比普通 RAQ 默认更高，但使用了 1.5 倍信道符号。旧 RVQ 曾经更好与当前更差可以同时成立。",
'',
'无法仅凭现存文件确定你当时所看的第一版是哪次具体结果；旧 RVQ 配置变化是有历史 JSON 和复测双重依据的解释。另有更早的 ch64-128 普通 RAQ 权重，不能把它与当前 ch256-512 默认权重视为同一模型。',
'',
'**统一信息量与链路后的结果**',
'',
'第一尺度为 32×32=1024 个索引，第二尺度为 16×16=256 个索引。普通 RAQ [8,16] 与 RVQ [[4,2],[8,2]] 不仅总 bit 相同，各尺度 bit 也完全相同：1024×3+256×4=4096 bit。',
'',
'| 条件 | 普通 RAQ [8,16] PSNR / MS-SSIM | RVQ [[4,2],[8,2]] PSNR / MS-SSIM | RAQ PSNR 优势 |',
'|---|---|---|---:|',
f"| SNR3、QPSK、LDPC0.5、单流、CBR1/48 | {ql['psnr']:.6f} / {ql['ms_ssim']:.6f} | {rl['psnr']:.6f} / {rl['ms_ssim']:.6f} | {diff['matched_low_channel_raq_minus_rvq_psnr']:.6f} dB |",
f"| 关闭信道、4096 bit/图 | {qnl['psnr']:.6f} / {qnl['ms_ssim']:.6f} | {rn['psnr']:.6f} / {rn['ms_ssim']:.6f} | {diff['matched_low_no_channel_raq_minus_rvq_psnr']:.6f} dB |",
'',
'6144 bit/图的补充对照也保持各尺度 bit 一致。它说明同等总 bit 下，RVQ 如何分配两级码本也会改变性能：',
'',
'| 6144 bit/图、无信道的配置 | PSNR | MS-SSIM |',
'|---|---:|---:|',
f"| 普通 RAQ [32,16] | {qn['psnr']:.6f} | {qn['ms_ssim']:.6f} |"]
for label in ['matched_high_8x4_no_channel','matched_high_4x8_no_channel','matched_high_16x2_no_channel']:
 r=get(rvq,label);lines.append(f"| RVQ {r['layout']} | {r['psnr']:.6f} | {r['ms_ssim']:.6f} |")
lines += [
'',
'这只是当前权重、这些具体配置和 Kodak24 上的结果，不能推出 RVQ 机制在所有码率下都无效。',
'',
'**训练与残差量化核查**',
'',
'RVQ 训练脚本固定验证配置是 [[16,16],[4,4]]，9216 bit/图；测试默认却为 4096 bit/图，信息量只有验证配置的 44.4%。best 的指标反映另一配置的表现，且验证包含源码本重建、RAQ重建及模拟信道，不是此处 Kodak 无信道 PSNR。',
'',
'课程列表相同也不意味着训练码率分布相同。普通 RAQ 每次采样两个 K；独立 RVQ 每次分别采样四个 K。在晚期的六种候选下，配置空间分别为 6²=36 与 6⁴=1296。在相同每图索引数、每级候选分布下，RVQ 平均信息 bit 是普通 RAQ 的两倍，当前包含 K=2 的测试配置仅在晚期课程中覆盖。低码率布局得到的训练和模型选择关注不足，是合理推测；没有重新训练验证它的因果贡献。',
'',
'历史训练日志显示两次有效总 batch 都是 24，但普通 RAQ 微 batch 12、RVQ 微 batch 24，实际训练 GPU 也不同。普通 RAQ 验证时未固定目标 K；RVQ 固定四 K。对应 CSV 均记录 200 epoch，最佳 val_recon 分别 0.03349796（epoch193）/0.02530692（epoch198）。这些验证值因配置不同不能直接判定哪个 Kodak 低码率结果更好。',
'',
'量化代码按 residual−quantized 逐级处理，重建端对各级特征求和，训练和测试都调用同一独立量化实现；两份 best 权重以严格方式加载了对应模块，源码本维度和网络规模均匹配，未发现错误切到其他 RVQ 模式或误用 res6/last 权重。',
'',
'当前 RVQ [[4,2],[8,2]] 的首尺度残差 MSE 从 0.194545 降至 0.151801；第二尺度却由 0.029128 增至 0.039946（约+37.14%）。代码没有保留零码字或保证残差误差单调下降的约束，所以小码本第二级可能增加特征误差。',
'',
f"但临时仅移除第二尺度第二级，图像 PSNR 从 {rn['psnr']:.6f} 降至 {rvq['drop_scale1_stage2_probe']['psnr']:.6f}，MS-SSIM 从 {rn['ms_ssim']:.6f} 变为 {rvq['drop_scale1_stage2_probe']['ms_ssim']:.6f}。这说明特征 MSE 变差不能直接等同于图像指标必然变差，也不能凭此认定实现有错误。此探针只改变内存中的解码输入，没有重训练或修改权重。",
'',
f"还测试了各自训练权重中的相同源码本 [64,64] 支路：普通 RAQ {get(raq,'source_no_channel')['psnr']:.6f} dB，RVQ {get(rvq,'source_no_channel')['psnr']:.6f} dB。RVQ 源支路并不更差，更支持问题集中在具体目标码本和残差分配，而非整个编码器/解码器都退化。两者都是联合训练模型中的源码本支路，不是独立训练的无 RAQ 消融。",
'',
'**相关文件**',
'',
f"- [第一版脚本]({ROOT}/scripts/eval/test_src64_64_raq2_64_curriculum_ch256_512.sh:20)",
f"- [RVQ 当前码本]({ROOT}/scripts/eval/test_independent_raq_rvq_src64_64_d2_combined.sh:32)",
f"- [RVQ 当前 LDPC 值]({ROOT}/scripts/eval/test_independent_raq_rvq_src64_64_d2_combined.sh:77)",
f"- [RVQ 训练验证布局]({ROOT}/scripts/train/current/run_independent_raq_rvq_src64_64_k2_64_d2_curriculum_ch256_512.sh:33)",
f"- [残差量化实现]({ROOT}/models/independent_raq_rvq.py:28)",
f"- [保留旧配置的结果 JSON]({ROOT}/experiments/eval/independent_raq_rvq_src64-64_k4x64-8x2_d2_ldpc34_qpsk_per_stage.json)",
f"- [普通 RAQ 本次全部实测]({OUT}/raq_diagnostics.json)",
f"- [RVQ 本次全部实测]({OUT}/independent_rvq_diagnostics.json)",
f"- [原目录未变验证]({OUT}/verification.json)",
'']
(OUT/'analysis.md').write_text('\n'.join(lines))
print(json.dumps({'report':str(OUT/'analysis.md'),'verification':verification,'summary':diff},ensure_ascii=False,indent=2))
