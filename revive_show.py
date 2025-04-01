# ------------------------------------------------------------------------------------------------------------
import gymnasium as gym
import onnxruntime
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation

def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save("./revive_result.gif", writer="pillow", fps = 30)

frames = []

policy_path = "/home/wei/my_d3/model.onnx"
policy = onnxruntime.InferenceSession(policy_path)

env = gym.make("Pendulum-v1",render_mode="rgb_array")
observation, _ = env.reset()
policy_output_names = ["actions"]

while True:
    frames.append(env.render())
    policy_input = {'states' : np.array(observation, dtype=np.float32).reshape(1, -1)}
    output = policy.run(input_feed=policy_input, output_names=policy_output_names)
    action = output[0][0]

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