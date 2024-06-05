from pymgrid.envs import DiscreteMicrogridEnv
from RL_custom_Reward import *

class CustomMicrogridEnv(DiscreteMicrogridEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_shaper = GridUsagePenaltyShaper()

    def step(self, action):
        observation, reward, done, info = super().step(action)

        # Get additional reward from the reward shaper
        reward = reward - self.reward_shaper(step_info=info)

        # Return the modified tuple
        return observation, reward, done, info
