import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# 1. 包含 SNR + 预算的数据集定义
# ==========================================
class CodebookDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # 1. 提取 SNR 并归一化 (-10 到 20)
        self.snr_min, self.snr_max = -10.0, 20.0
        snr_values = df['SNR'].values
        norm_snr = (snr_values - self.snr_min) / (self.snr_max - self.snr_min)
        self.snr = torch.tensor(norm_snr, dtype=torch.float32).unsqueeze(1)

        # 2. 提取预算范围并归一化 (除以 1000000)
        df[['Budget_Min', 'Budget_Max']] = df['Bit_Budget_Range'].str.split('-', expand=True).astype(float)
        self.budget_min = torch.tensor(df['Budget_Min'].values, dtype=torch.float32).unsqueeze(1) / 1000000.0
        self.budget_max = torch.tensor(df['Budget_Max'].values, dtype=torch.float32).unsqueeze(1) / 1000000.0

        # 拼接 3 个特征作为输入: [Batch, 3]
        self.X = torch.cat([self.snr, self.budget_min, self.budget_max], dim=1)

        # 3. 提取标签 (K1, K2, K3, K4)，转换为 0~7 的类别索引
        k_values = df[['K1', 'K2', 'K3', 'K4']].values
        k_indices = np.log2(k_values) - 5
        self.Y = torch.tensor(k_indices, dtype=torch.long)  # [Batch, 4]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ==========================================
# 2. 三输入 MLP 神经网络架构 (input_dim=3)
# ==========================================
class CodebookPredictorMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=4, num_classes=8):
        super(CodebookPredictorMLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim))  # 加点 LayerNorm 让多特征融合更稳

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))

        self.backbone = nn.Sequential(*layers)

        self.head1 = nn.Linear(hidden_dim, num_classes)
        self.head2 = nn.Linear(hidden_dim, num_classes)
        self.head3 = nn.Linear(hidden_dim, num_classes)
        self.head4 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out1 = self.head1(features)
        out2 = self.head2(features)
        out3 = self.head3(features)
        out4 = self.head4(features)
        return torch.stack([out1, out2, out3, out4], dim=1)


# ==========================================
# 3. 训练流程
# ==========================================
def train():
    set_seed(42)
    csv_path = "/home/yi/wk-1/vq-dc-raqmae-64-transformer/dataset.csv"
    batch_size = 32
    learning_rate = 2e-3
    epochs = 5000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = CodebookDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CodebookPredictorMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"开始训练全能 MLP (SNR+Budget -> Codebooks)，使用设备: {device}...")

    # 【新增】：初始化一个正无穷大的最佳 loss，用于记录历史最低点
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = sum([criterion(outputs[:, i, :], batch_y[:, i]) for i in range(4)])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # 计算当前 epoch 的平均 loss
        avg_loss = total_loss / len(dataloader)

        # 【核心修改】：只有当当前 loss 创新低时，才保存权重！
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "best_mlp_codebook.pth")
        # 每 50 轮打印一次进度，并汇报当前的最佳战绩
        if epoch % 50 == 0:
            print(
                f"Epoch [{epoch}/{epochs}] | 当前 Loss: {avg_loss:.4f} | 历史最佳 Loss: {best_loss:.4f} (在第 {best_epoch} 轮)")
    print(f"\n✅ 训练完成！历史最低 Loss 为 {best_loss:.4f}。")
    print(f"💾 最优模型权重 (第 {best_epoch} 轮) 已永久保存至: best_mlp_codebook.pth")

if __name__ == "__main__":
    train()
