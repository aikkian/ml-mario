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

import re

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace


# The level we want to beat first. "v0" includes the standard reward function
# (reward for moving right + reaching the flag; penalty for dying / wasting time).
DEFAULT_LEVEL = "SuperMarioBros-1-1-v0"

# Map the "vN" suffix in a level id to nes-py's rom_mode (v0 = the normal game).
_ROM_MODES = {"0": "vanilla", "1": "downsample", "2": "pixel", "3": "rectangle"}


def _make_base_env(level):
    """Create the raw Mario env and reduce the controls to a small, sensible set.

    We build the environment DIRECTLY instead of via ``gym_super_mario_bros.make``.
    Why: gym 0.26 wraps ``make()`` output in helper wrappers (TimeLimit,
    OrderEnforcing) that assume the NEW 5-value step API, but nes-py still uses
    the OLD 4-value API — which crashes with
    "not enough values to unpack (expected 5, got 4)". Constructing the env class
    directly skips those wrappers; our MarioGymnasium adapter then handles the
    old API itself.

    SIMPLE_MOVEMENT is ~7 button combos (e.g. "run right", "jump right") instead
    of every possible NES input. Fewer choices = much faster learning.
    """
    match = re.match(r"SuperMarioBros(2?)-(\d+)-(\d+)-v(\d)", level)
    try:
        if match:
            lost_levels = match.group(1) == "2"
            world, stage, version = int(match.group(2)), int(match.group(3)), match.group(4)
            env = SuperMarioBrosEnv(
                rom_mode=_ROM_MODES.get(version, "vanilla"),
                lost_levels=lost_levels,
                target=(world, stage),
            )
        else:
            env = SuperMarioBrosEnv()
    except Exception:
        # Fallback: the registered make() path (used by the older gym combo).
        try:
            env = gym_super_mario_bros.make(level, disable_env_checker=True)
        except TypeError:
            env = gym_super_mario_bros.make(level)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


class MarioGymnasium(gymnasium.Env):
    """Wraps the old-style Mario env in the modern gymnasium API AND applies all
    of our observation preprocessing."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, level=DEFAULT_LEVEL, skip=4, stack=4,
                 width=84, height=84, render_mode=None, stuck_steps=200):
        super().__init__()
        self._env = _make_base_env(level)

        self._skip = skip          # how many frames each action is held for
        self._stack = stack        # how many recent frames the agent sees at once
        self._width = width
        self._height = height
        self.render_mode = render_mode
        self._frames = deque(maxlen=stack)

        # STUCK DETECTION: if Mario makes no rightward progress for this many
        # steps (e.g. jammed against a pipe), we end the episode early instead of
        # waiting out the in-game timer. This avoids wasting training time on
        # dead attempts. 0 disables it. (Each step is `skip` frames, so 200 steps
        # ~= 800 frames ~= 13 seconds of no progress.)
        self._stuck_steps = stuck_steps
        self._max_x = 0            # furthest right Mario has reached this episode
        self._stuck_counter = 0    # steps since that furthest point

        # The most recent FULL-COLOR game frame (before grayscale/resize). Kept
        # so play.py can record nice-looking videos. Shape ~ (240, 256, 3) uint8.
        self._last_rgb = None

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
        self._last_rgb = obs       # keep the full-color frame for recording
        frame = self._preprocess(obs)
        # Fill the whole stack with the first frame to start.
        for _ in range(self._stack):
            self._frames.append(frame)
        # Reset the stuck tracker for the new attempt.
        self._max_x = 0
        self._stuck_counter = 0
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

        self._last_rgb = obs       # keep the full-color frame for recording
        frame = self._preprocess(obs)
        self._frames.append(frame)

        # gymnasium splits "episode over" into terminated (a real game-over: Mario
        # died or reached the flag) vs truncated (the episode was cut short for
        # another reason). The Mario env reports game-overs via `done`.
        terminated = done
        truncated = False

        # STUCK DETECTION: end the episode early if Mario stops advancing right.
        if self._stuck_steps:
            x_pos = info.get("x_pos")
            if x_pos is not None:
                if x_pos > self._max_x:
                    self._max_x = x_pos       # new furthest point -> not stuck
                    self._stuck_counter = 0
                else:
                    self._stuck_counter += 1  # no progress this step
                if self._stuck_counter >= self._stuck_steps:
                    # Cut it short. This is "truncated", not "terminated",
                    # because the game didn't actually end — which is the
                    # correct signal for the learning algorithm.
                    truncated = True

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


def make_mario_env(level=DEFAULT_LEVEL, render_mode=None, stuck_steps=200):
    """Convenience factory used by train.py and play.py.

    `stuck_steps` ends an episode early after that many steps without rightward
    progress (set 0 to disable). This speeds up training by not wasting frames on
    a jammed Mario.
    """
    return MarioGymnasium(level=level, render_mode=render_mode,
                          stuck_steps=stuck_steps)
