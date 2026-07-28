import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_train import CodebookPredictorMLP


# ==========================================
# 2. 加载权重并进行连续预测绘图
# ==========================================
@torch.no_grad()
def load_and_plot(weight_path, test_b_min, test_b_max):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 实例化模型并加载权重
    model = CodebookPredictorMLP().to(device)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件 {weight_path}，请先运行 train_mlp.py")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print(f"✅ 成功加载预训练权重: {weight_path}")

    # 构造连续的 SNR 序列
    snr_min, snr_max = -10.0, 20.0
    continuous_snrs = np.linspace(snr_min, snr_max, 3000)
    norm_snrs = (continuous_snrs - snr_min) / (snr_max - snr_min)
    snr_tensor = torch.tensor(norm_snrs, dtype=torch.float32).unsqueeze(1)

    # 构造固定的预算序列
    b_min_norm = test_b_min / 1000000.0
    b_max_norm = test_b_max / 1000000.0
    b_min_tensor = torch.full((3000, 1), b_min_norm, dtype=torch.float32)
    b_max_tensor = torch.full((3000, 1), b_max_norm, dtype=torch.float32)

    # 拼接得到完整的输入 [3000, 3]
    x_tensor = torch.cat([snr_tensor, b_min_tensor, b_max_tensor], dim=1).to(device)

    # 批量预测
    outputs = model(x_tensor)
    predicted_indices = torch.argmax(outputs, dim=2).cpu().numpy()

    # 将索引转回真实 K 值
    k_predictions = 2 ** (predicted_indices + 5)

    # === 开始绘制图表 (修改为 2x2 子图) ===
    # 创建一个 2 行 2 列的画布，调整大小让每个子图都有充足的显示空间
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()  # 摊平为一维列表，方便用 for 循环遍历

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['K1 (Layer 1)', 'K2 (Layer 2)', 'K3 (Layer 3)', 'K4 (Layer 4)']
    y_ticks = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    y_tick_labels = ['32', '64', '128', '256', '512', '1024', '2048', '4096']

    for i in range(4):
        ax = axes[i]

        # 在对应的子图上绘制阶梯图
        ax.step(continuous_snrs, k_predictions[:, i], where='mid',
                color=colors[i], linewidth=2.5, alpha=0.8)

        # 为每个子图单独设置标题、坐标轴和网格
        ax.set_title(f"{labels[i]} vs SNR", fontsize=14, fontweight='bold', color=colors[i])
        ax.set_xlabel("Continuous SNR (dB)", fontsize=12)
        ax.set_ylabel("Target Codebook Size (K)", fontsize=12)
        ax.set_yscale('log', base=2)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    # 设置整个大图的全局标题
    fig.suptitle(f"Predicted Codebook Sizes vs Continuous SNR\n(Budget Constraints: {test_b_min} - {test_b_max} Bits)",
                 fontsize=18, fontweight='bold', y=1.02)

    # 自动调整子图间距，防止文字重叠
    plt.tight_layout()

    save_name = f"continuous_snr_budget_{test_b_min}_{test_b_max}_separated.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"📊 绘图完成！独立子图版已保存为: {save_name}")


if __name__ == "__main__":
    WEIGHT_FILE = "/home/yi/wk-1/vq-dc-raqmae-64-transformer/MLP/best_mlp_codebook.pth"
    TEST_BUDGET_MIN = 700000
    TEST_BUDGET_MAX = 1800000

    # 直接加载权重画图，无需训练
    load_and_plot(WEIGHT_FILE, TEST_BUDGET_MIN, TEST_BUDGET_MAX)