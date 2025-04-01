import d3rlpy
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation

def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save("./my_result.gif", writer="pillow", fps = 30)
    
frames = []
# 加载训练好的模型
pendulum = d3rlpy.load_learnable("/home/wei/my_d3/d3rlpy_logs/CQL_20250330144540/model_20000.d3")

# 创建环境
env = gym.make("Pendulum-v1",render_mode="rgb_array")

# 获取初始观察值
observation, _ = env.reset()

while True:
    # 使用 predict 方法进行推断
    frames.append(env.render())
    action = pendulum.predict(np.expand_dims(observation, axis=0))[0]
    # 执行动作
    observation, reward, done, truncated, info = env.step(action)
    print(observation, reward, done, truncated, info, action)
    # 如果环境结束，退出循环
    if truncated:
        print("ok")
        break

# 关闭环境
env.close()
display_frames_as_gif(frames)
# # ------------------------------------------------------------------------------------------------------------
# import gymnasium as gym
# import onnxruntime
# import numpy as np
# import pickle

# policy_path = "/home/wei/my_d3/policy.pkl"
# # policy = onnxruntime.InferenceSession(policy_path)
# policy = pickle.load(open(policy_path, 'rb'), encoding='utf-8')

# env = gym.make("Pendulum-v1",render_mode="human")
# observation, _ = env.reset()
# print(observation)
# frames = []
# policy_output_names = ["actions"]
# cnt = 0

# while True:
#     policy_input = {'states' : np.array(observation, dtype=np.float32).reshape(1, -1)}
#     output = policy.run(input_feed=policy_input, output_names=policy_output_names)
#     action = output[0][0]

#     # 执行动作
#     observation, reward, done, truncated, info = env.step(action)
#     print(observation, reward, done, truncated, info, action)
    
#     # 如果环境结束，退出循环
#     if truncated:
#         print("ok")
#         break

# # 关闭环境
# env.close()