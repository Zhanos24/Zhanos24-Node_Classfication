# Bryantzw keep focus keep trying
# time:2023/12/26
import os
import matplotlib.pyplot as plt

# 获取数据文件路径
data_dir = "../data/EI_DATA"
data_files = ["imdbnewma.txt", "dblpnewma.txt", "alibabatestma.txt"]
data_labels = ["IMDB", "DBLP", "Alibaba"]

# 创建空列表来存储所有数据
all_data = []

# 逐个读取数据文件并添加到 all_data 列表
for file in data_files:
    data_file = os.path.join(data_dir, file)
    with open(data_file, 'r') as f:
        data = [float(line.strip()) for line in f.readlines()]
        all_data.append(data)

# 设置折线图的参数
colors = ['blue', 'red', 'green']
markers = ['^', 'o', 's']
marker_freq = 20
line_width = 0.8
plt.figure(dpi=250)
# 绘制折线图
for i, data in enumerate(all_data):
    # x = [j + 1 for j in range(len(data)) if (j + 1) % 1 == 0]
    # y = [data[j - 1] for j in x]
    x = []
    y = []

    # 仅保留第5或者5的倍数轮次的数据
    for j, val in enumerate(data):
        if (j + 1) % 1 == 0:
            x.append(j + 1)
            y.append(val)
    # 只使用颜色来区分折线
    plt.plot(x, y, color=colors[i], linewidth=line_width, label=data_labels[i])
    # 在每20轮处添加标记（实心）
    for j, val in enumerate(x):
        if (val - x[0]) % marker_freq == 0:
            plt.plot(val, y[j], marker=markers[i], markersize=3, markerfacecolor=colors[i], markeredgewidth=0, color=colors[i])

# 添加标签
plt.xlabel("rounds")
plt.ylabel("F1-Macro")

# 添加图例
# plt.legend(loc='best', fontsize='small', shadow=False, framealpha=1)

# 设置图像分辨率为300dpi
# plt.savefig("plotma.png", dpi=300)

# 显示图表
plt.show()