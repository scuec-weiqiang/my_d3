import d3rlpy

# 加载数据集（使用随机提取器）
dataset, env = d3rlpy.datasets.get_pendulum(
    transition_picker=['random'])

# 查看提取的单步转移
for i in range(3):
    transition = dataset[i]  # 随机抽取的 (s, a, r, s')
    print(f"Transition {i}:", transition)