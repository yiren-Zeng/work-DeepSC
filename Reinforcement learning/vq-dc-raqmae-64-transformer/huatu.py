import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
df = pd.read_csv('/home/yi/wk-1/vq-dc-raqmae-64-transformer/optimal_raq_codebook_rl_dataset.csv')

# 第一张图：Z轴为 K1
fig, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
ax1.scatter(df['SNR'], df['Bit_Budget'], df['K1'], c='r', marker='o', alpha=0.5)
ax1.set_xlabel('SNR')
ax1.set_ylabel('Bit Budget')
ax1.set_zlabel('K1')
ax1.set_title('SNR vs Bit_Budget vs K1')
plt.tight_layout()
plt.savefig('plot_k1.png')
plt.close(fig)

# 第二张图：Z轴为 K2
fig, ax2 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
ax2.scatter(df['SNR'], df['Bit_Budget'], df['K2'], c='g', marker='^', alpha=0.5)
ax2.set_xlabel('SNR')
ax2.set_ylabel('Bit Budget')
ax2.set_zlabel('K2')
ax2.set_title('SNR vs Bit_Budget vs K2')
plt.tight_layout()
plt.savefig('plot_k2.png')
plt.close(fig)

# 第三张图：Z轴为 K3
fig, ax3 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
ax3.scatter(df['SNR'], df['Bit_Budget'], df['K3'], c='b', marker='s', alpha=0.5)
ax3.set_xlabel('SNR')
ax3.set_ylabel('Bit Budget')
ax3.set_zlabel('K3')
ax3.set_title('SNR vs Bit_Budget vs K3')
plt.tight_layout()
plt.savefig('plot_k3.png')
plt.close(fig)

# 第四张图：Z轴为 K4
fig, ax4 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
ax4.scatter(df['SNR'], df['Bit_Budget'], df['K4'], c='m', marker='*', alpha=0.5)
ax4.set_xlabel('SNR')
ax4.set_ylabel('Bit Budget')
ax4.set_zlabel('K4')
ax4.set_title('SNR vs Bit_Budget vs K4')
plt.tight_layout()
plt.savefig('plot_k4.png')
plt.close(fig)