# import numpy as np

# dataset = np.load("/home/wei/my_d3/expert_data_with_next.npz")
# print(dataset)
# states = dataset["states"]
# actions = dataset["actions"]
# index = dataset["index"]
# next_states = dataset["next_states"]
# # next_states = np.zeros_like(states)

# print(states)
# print(actions)
# print(index)
# print(next_states)

# import numpy as np

# dataset = np.load("/home/wei/my_d3/expert_data.npz")
# states = dataset["states"]
# index = dataset["index"]

# # 初始化 next_state 为全零
# next_states = np.zeros_like(states)

# # 通过 index 确定轨迹边界
# traj_boundaries = np.concatenate([[0], index])  # 添加起始点 0

# # 批量处理每个轨迹
# for i in range(len(traj_boundaries) - 1):
#     start = traj_boundaries[i]
#     end = traj_boundaries[i+1]
    
#     # 直接操作数组切片
#     next_states[start:end-1] = states[start+1:end]  # 最后一个状态自动保持零

# # 验证数据一致性
# assert len(states) == traj_boundaries[-1], "轨迹总长度与 states 维度不匹配"

# # 更新数据集
# new_dataset = {**dataset, "next_states": next_states}

# # 可选：保存新数据集
# np.savez("/home/wei/my_d3/expert_data_with_next.npz", **new_dataset)

# print("Next_state 已成功添加，轨迹边界已验证。")