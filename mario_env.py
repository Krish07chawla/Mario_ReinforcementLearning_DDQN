import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import SuperMarioBrosEnv


class SkipFrame(gym.Wrapper):
    """Repeat the same action for `skip` frames and add up the rewards."""
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayResize(gym.ObservationWrapper):
    """RGB (240,256,3) -> grayscale -> (84,84), uint8."""
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)


class FrameStack(gym.Wrapper):
    """Keep the last n frames as one array (n,84,84)."""
    def __init__(self, env, n):
        super().__init__(env)
        self.n = n
        self.frames = deque(maxlen=n)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n):
            self.frames.append(obs)
        return np.stack(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames), reward, terminated, truncated, info


def make_env(skip=4):
    """Returns (wrapped_env, base_env). base_env is used to grab RGB screens for GIFs."""
    base = SuperMarioBrosEnv(rom_mode="vanilla", target=(1, 1))
    env = JoypadSpace(base, [["right"], ["right", "A"]])   # 0 = walk right, 1 = jump right
    env = SkipFrame(env, skip=skip)
    env = GrayResize(env, size=84)
    env = FrameStack(env, n=4)
    return env, base
