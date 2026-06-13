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
        truncated = False
        max_x = 0          # furthest right Mario reached this attempt

        while not done:
            env.render()
            # deterministic=True -> take the agent's single best action (no random
            # exploration), so we see its actual skill.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            max_x = max(max_x, info.get("x_pos", 0))
            if info.get("flag_get"):
                beat_level = True

        # Explain how the attempt ended: beat the flag, got stuck (our early
        # cut-off truncates with no game-over), or actually died / ran out of time.
        if beat_level:
            status = "🏁 BEAT THE LEVEL!"
        elif truncated:
            status = "stuck (no progress)"
        else:
            status = "died / out of time"
        print(
            f"Episode {episode}: reward={total_reward:.0f}  "
            f"reached x={max_x}  ->  {status}"
        )

    env.close()


if __name__ == "__main__":
    main()
