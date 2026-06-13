"""
smoke_test.py
=============

A "smoke test" is a tiny check that the basics work BEFORE you invest in the
real thing. Run this first. It does NOT do any machine learning — it just:

  1. opens the Mario game,
  2. makes Mario take random actions for a few seconds,
  3. confirms the observation shape is what we expect,

so you can be confident the emulator + libraries installed correctly. If this
runs without errors, you're ready to train. If it fails, see the README
"Troubleshooting" section (this is almost always a dependency version issue).

Run it:
    python src/smoke_test.py
"""

import _compat  # noqa: F401  (Python 3.13 / NumPy 2.x shims — keep first)

from env import make_mario_env


def main():
    # Open a window so you can see Mario moving around randomly.
    env = make_mario_env(render_mode="human")

    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}   (expected: (4, 84, 84))")
    print(f"Number of possible actions: {env.action_space.n}")
    print("Running 500 random steps — you should see Mario flailing about...\n")

    for step in range(500):
        env.render()
        action = env.action_space.sample()           # a random button press
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()                     # start over if Mario dies

    env.close()
    print("\n✅ Smoke test finished with no errors. You're ready to train!")
    print("Next:  python src/train.py --timesteps 50000")


if __name__ == "__main__":
    main()
