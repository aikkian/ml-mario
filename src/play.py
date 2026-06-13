"""
play.py
=======

This is the payoff: load a brain you trained and WATCH Mario play in a window,
using its learned best moves instead of random ones.

Run it on different saved checkpoints to literally see it improve over training:
flailing  ->  walking right  ->  jumping the first Goomba  ->  reaching the flag.

Run it:
    python src/play.py --model mario_ppo_final
    python src/play.py --model models/mario_ppo_500000_steps --episodes 3
"""

import _compat  # noqa: F401  (Python 3.13 / NumPy 2.x shims — keep first)

import argparse

from stable_baselines3 import PPO

from env import make_mario_env


def main():
    parser = argparse.ArgumentParser(description="Watch a trained agent play Mario.")
    parser.add_argument(
        "--model", type=str, default="mario_ppo_final",
        help="Path to a saved model (without the .zip extension is fine).",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="How many full attempts to watch.",
    )
    args = parser.parse_args()

    # render_mode="human" opens the game window so you can see it.
    env = make_mario_env(render_mode="human")
    model = PPO.load(args.model)

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        beat_level = False

        while not done:
            env.render()
            # deterministic=True -> take the agent's single best action (no random
            # exploration), so we see its actual skill.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            if info.get("flag_get"):
                beat_level = True

        status = "🏁 BEAT THE LEVEL!" if beat_level else "died / timed out"
        print(f"Episode {episode}: reward={total_reward:.0f}  ->  {status}")

    env.close()


if __name__ == "__main__":
    main()
