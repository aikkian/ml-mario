"""
env.py
======

This file builds the *environment* — the world the agent lives in. In
reinforcement learning the "environment" is the game plus everything that
defines how the agent perceives it and what rewards it earns.

Two jobs happen here:

1. ADAPTER: The Mario library (`gym-super-mario-bros`) was written for the OLD
   `gym` interface. Modern Stable-Baselines3 expects the NEW `gymnasium`
   interface. The `MarioGymnasium` class below is a thin translator so the two
   can talk to each other. (This is the #1 thing beginners get stuck on.)

2. PERCEPTION ("observation engineering"): The raw game screen is big and
   colorful, which makes learning slow. We:
     - convert it to grayscale (color isn't needed to play),
     - shrink it to 84x84 pixels,
     - skip frames (hold each action for a few frames),
     - stack the last 4 frames together so the agent can perceive MOTION
       (which way Mario and the enemies are moving) from still images.

An agent is only as smart as what it can perceive, so this step matters a lot.
"""

# Compatibility shims for Python 3.13 / NumPy 2.x — MUST be imported first.
import _compat  # noqa: F401

from collections import deque

import cv2
import numpy as np
import gymnasium
from gymnasium import spaces

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


# The level we want to beat first. "v0" includes the standard reward function
# (reward for moving right + reaching the flag; penalty for dying / wasting time).
DEFAULT_LEVEL = "SuperMarioBros-1-1-v0"


def _make_base_env(level):
    """Create the raw Mario env and reduce the controls to a small, sensible set.

    SIMPLE_MOVEMENT is ~7 button combos (e.g. "run right", "jump right") instead
    of every possible NES input. Fewer choices = much faster learning.
    """
    try:
        # Newer gym needs the env checker disabled so it doesn't reject the old
        # Mario library's return format.
        env = gym_super_mario_bros.make(level, disable_env_checker=True)
    except TypeError:
        # Older gym (the fallback combo) doesn't know that argument.
        env = gym_super_mario_bros.make(level)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


class MarioGymnasium(gymnasium.Env):
    """Wraps the old-style Mario env in the modern gymnasium API AND applies all
    of our observation preprocessing."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, level=DEFAULT_LEVEL, skip=4, stack=4,
                 width=84, height=84, render_mode=None):
        super().__init__()
        self._env = _make_base_env(level)

        self._skip = skip          # how many frames each action is held for
        self._stack = stack        # how many recent frames the agent sees at once
        self._width = width
        self._height = height
        self.render_mode = render_mode
        self._frames = deque(maxlen=stack)

        # ACTION SPACE: a discrete number of button combos (carried over from
        # JoypadSpace). e.g. action 1 might mean "press right".
        self.action_space = spaces.Discrete(self._env.action_space.n)

        # OBSERVATION SPACE: `stack` grayscale frames of size height x width,
        # pixel values 0-255. Shape (4, 84, 84) is "channels-first", which is
        # exactly what Stable-Baselines3's image policy (CnnPolicy) expects.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(stack, height, width), dtype=np.uint8
        )

    # -- helpers --------------------------------------------------------------

    def _preprocess(self, frame):
        """Turn one raw RGB screen into a small grayscale image."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            gray, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        return resized.astype(np.uint8)

    def _stacked(self):
        """Combine the last few frames into one observation of shape (stack, H, W)."""
        return np.array(self._frames, dtype=np.uint8)

    # -- gymnasium API --------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Start a fresh episode (a new attempt at the level)."""
        super().reset(seed=seed)
        obs = self._env.reset()
        # Old gym returns just `obs`; newer gym returns `(obs, info)`.
        if isinstance(obs, tuple):
            obs = obs[0]
        frame = self._preprocess(obs)
        # Fill the whole stack with the first frame to start.
        for _ in range(self._stack):
            self._frames.append(frame)
        return self._stacked(), {}

    def step(self, action):
        """Apply one action (held for `skip` frames) and return what happened."""
        total_reward = 0.0
        done = False
        info = {}
        obs = None
        for _ in range(self._skip):
            result = self._env.step(action)
            # Defensive: handle BOTH the old 4-tuple and the new 5-tuple format,
            # so this works regardless of which gym version is installed.
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = bool(terminated) or bool(truncated)
            else:
                obs, reward, done, info = result
            total_reward += reward
            if done:
                break

        frame = self._preprocess(obs)
        self._frames.append(frame)

        # gymnasium splits "episode over" into terminated (e.g. died / won) vs
        # truncated (time limit). The Mario env lumps these together, so we
        # report it all as `terminated`.
        terminated = done
        truncated = False
        return self._stacked(), float(total_reward), terminated, truncated, info

    def render(self):
        """Open/refresh the game window (used when watching it play)."""
        try:
            return self._env.render()
        except TypeError:
            # Older nes-py expects a mode argument.
            return self._env.render(mode="human")

    def close(self):
        self._env.close()


def make_mario_env(level=DEFAULT_LEVEL, render_mode=None):
    """Convenience factory used by train.py and play.py."""
    return MarioGymnasium(level=level, render_mode=render_mode)
