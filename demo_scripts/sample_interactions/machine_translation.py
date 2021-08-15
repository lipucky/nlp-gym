import os
import re

from nlp_gym.data_pools.base import Sample
from nlp_gym.envs.quality_estimation.env import NeuralMachineTranslationEnv
from nlp_gym.envs.quality_estimation.reward import SimpleBLEUReward

with open(os.path.expanduser("~/KiwiCutter/WMT19/small/train.src"), "r") as f:
    content = f.read()
    wmt_data_en = content.splitlines()

with open(os.path.expanduser("~/KiwiCutter/WMT19/small/train.pe"), "r") as f:
    content = f.read()
    wmt_data_de = content.splitlines()

env = NeuralMachineTranslationEnv(corpus=wmt_data_de, reward_function=SimpleBLEUReward())

for en_sent, de_sent in zip(wmt_data_en, wmt_data_de):
    de_sent = de_sent.lower()
    env.add_sample(Sample(en_sent, re.findall(r"\w+|[^\w\s]", de_sent, re.UNICODE)))

done = False
state = env.reset()
total_reward = 0
for i in range(10):
    print(env.vocab)
    env.render()
    user_input = input("Enter next word: ")
    if user_input in env.vocab:
        action = env.action_space.action_to_ix(user_input)
    else:
        action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    total_reward += reward
    print(f"Action: {env.action_space.ix_to_action(action)}")
print(f"Episodic reward {total_reward}")
