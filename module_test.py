import torch
import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 加载环境模型
model = torch.load("/home/wei/my_d3/logs/model900.pth", weights_only=False)
model.eval()  # 设置模型为评估模式

# 选择设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # 将模型转移到设备上

env = gym.make('Pendulum-v1')

# 初始化状态
# theta = np.pi / 4  # 初始角度
# theta_dot = 0  # 初始角速度
# state = np.array([theta, theta_dot, 0])  # 假设加入额外的状态信息（如加速度和扭矩）
state ,_ = env.reset()
# 时间步长和模拟时长
dt = 0.05  # 时间步长
t_max = 10  # 总时间
num_steps = int(t_max / dt)

# 存储结果的列表
results = []

# 初始化gym环境
env.reset()

# 动画的设置
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Pendulum的线条和圆点
line, = ax.plot([], [], 'o-', lw=2)
line_gym, = ax.plot([], [], 'o-', lw=2, color='r')  # gym的Pendulum
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

# 动画更新函数
def update_frame(frame):
    global state
    
    # 在每个时间步生成一个相同的随机动作
    action = np.random.uniform(-2.0, 2.0, size=(1,))  # 假设动作范围是[-2, 2]

    # 使用相同的动作更新模型的状态
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # 状态转换为tensor
    action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)  # 动作转换为tensor
    
    # 使用模型进行预测（状态更新）
    with torch.no_grad():
        next_state_model = model(state_tensor, action_tensor).cpu().numpy().flatten()  # 转回 CPU 以便处理

    # 更新Gym环境的状态
    next_state_gym, _, _, _, _ = env.step(action)  # 使用相同的动作更新gym环境
    env.render()

    # 将模型和gym的角度转换为坐标
    x_model = np.sin(next_state_model[0])
    y_model = -np.cos(next_state_model[0])
    x_gym = np.sin(next_state_gym[0])
    y_gym = -np.cos(next_state_gym[0])

    # 更新Pendulum图形
    line.set_data([0, x_model], [0, y_model])
    line_gym.set_data([0, x_gym], [0, y_gym])

    # 更新时间文本
    time_text.set_text(f'Time: {frame*dt:.2f} s')

    # 将结果添加到表格中
    results.append({
        'time_step': frame * dt,
        'theta_model': next_state_model[0],
        'theta_dot_model': next_state_model[1],
        'theta_gym': next_state_gym[0],
        'theta_dot_gym': next_state_gym[1]
    })

    # 更新状态
    state = next_state_gym  # 更新模型的状态，确保两者一致

    # 停止条件：当一个完整循环结束时停止动画
    if frame >= num_steps:
        ani.event_source.stop()  # 停止动画更新
        plt.close()  # 关闭窗口

    return line, line_gym, time_text

# 创建动画并只运行一次循环
ani = FuncAnimation(fig, update_frame, frames=num_steps, interval=dt*1000, repeat=False)

# 保存为GIF文件
ani.save("pendulum_animation.gif", writer="imagemagick", fps=30)

# 显示图形
plt.show()

# 关闭gym环境
env.close()

# 将结果转换为DataFrame
df = pd.DataFrame(results)

# 输出结果表格
print(df)

# 保存为CSV文件
df.to_csv("pendulum_comparison.csv", index=False)
