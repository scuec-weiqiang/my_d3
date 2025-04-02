import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('pendulum_comparison.csv')

# 创建两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# 第一个图：绘制theta的变化
ax1.plot(df['time_step'], df['theta_model'], label='Theta (Model)', color='b', linestyle='-', marker='o')
ax1.plot(df['time_step'], df['theta_gym'], label='Theta (Gym)', color='r', linestyle='-', marker='x')
ax1.set_title('Theta Comparison')
ax1.set_xlabel('Time Step (s)')
ax1.set_ylabel('Theta (rad)')
ax1.legend()
ax1.grid(True)

# 第二个图：绘制theta_dot的变化
ax2.plot(df['time_step'], df['theta_dot_model'], label='Theta_dot (Model)', color='g', linestyle='-', marker='^')
ax2.plot(df['time_step'], df['theta_dot_gym'], label='Theta_dot (Gym)', color='purple', linestyle='-', marker='s')
ax2.set_title('Theta_dot Comparison')
ax2.set_xlabel('Time Step (s)')
ax2.set_ylabel('Theta_dot (rad/s)')
ax2.legend()
ax2.grid(True)

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()
