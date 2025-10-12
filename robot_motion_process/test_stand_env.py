import gymnasium as gym
from humanoidverse.envs.stand_task.humanoid_stand_env import HumanoidStandEnv
import numpy as np

env = HumanoidStandEnv(render_mode="human")
obs, _ = env.reset()
ret = 0.0
for t in range(300):
    action = np.zeros(env.action_space.shape, dtype=np.float32)  # 零动作：看是否能稳定站立
    obs, r, term, trunc, info = env.step(action)
    ret += r
    env.render()
    if term or trunc:
        break
print("Return:", ret)
env.close()
