import d3rlpy
import gym

env = gym.make("Pendulum-v1")

# prepare pretrained algorithm
sac = d3rlpy.load_learnable("/home/wei/test/d3rlpy_logs/SAC_20250326172407/model_10000.d3")

# prepare experience replay buffer
buffer = d3rlpy.dataset.create_infinite_replay_buffer()

# start data collection
sac.collect(env, buffer, n_steps=100000)

# save ReplayBuffer
with open("trained_policy_dataset.h5", "w+b") as f:
  buffer.dump(f)