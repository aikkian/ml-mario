"""
train.py
========

This is the heart of the project: the TRAINING LOOP. Run this to make Mario
learn.

What happens when you run it:
  1. We build the wrapped Mario environment (from env.py).
  2. We create a PPO agent with a CnnPolicy — a small neural network that looks
     at the screen images and decides which button to press.
  3. `model.learn(...)` runs the loop millions of times: the agent acts, sees
     the reward, and nudges its network to make rewarding actions more likely.
  4. Along the way we save checkpoints and watch for the level being beaten.

Run it:
    python src/train.py                  # full run (millions of steps, hours)
    python src/train.py --timesteps 50000  # quick test that everything works

Watch progress live (in a second terminal):
    tensorboard --logdir logs
then open the URL it prints (usually http://localhost:6006).
"""

import _compat  # noqa: F401  (Python 3.13 / NumPy 2.x shims — keep first)

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import make_mario_env
from callbacks import make_checkpoint_callback, FlagCallback


def build_training_env():
    """Wrap our Mario env so SB3 can train on it.

    - Monitor records episode rewards/lengths (used by the live charts).
    - DummyVecEnv lets SB3 treat it as a (here, single) batch of environments.
    """
    def _make():
        env = make_mario_env(render_mode=None)  # no window while training = faster
        return Monitor(env)

    return DummyVecEnv([_make])


def main():
    parser = argparse.ArgumentParser(description="Train an RL agent to play Mario.")
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="How many frames to train for. Start small (e.g. 50000) to test, "
             "then go big (1000000+) for real learning.",
    )
    parser.add_argument(
        "--save-name", type=str, default="mario_ppo_final",
        help="Filename (without extension) for the final saved model.",
    )
    args = parser.parse_args()

    env = build_training_env()

    # Create the PPO agent.
    #   "CnnPolicy" = the brain is a convolutional neural network that reads the
    #                 screen images.
    #   The hyperparameters below are sensible starting points for Mario. These
    #   are the knobs you'll experiment with later (see the README).
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log="logs",
        learning_rate=1e-4,   # how big each learning step is
        n_steps=512,          # frames collected before each network update
        batch_size=64,
        n_epochs=10,
        gamma=0.9,            # how much the agent values future vs immediate reward
        gae_lambda=1.0,
        ent_coef=0.01,        # encourages exploration (trying new things)
    )

    # Callbacks: save checkpoints + announce/save when the level is beaten.
    callbacks = [
        make_checkpoint_callback(save_dir="models", save_freq=50_000),
        FlagCallback(save_dir="models"),
    ]

    print(f"Training for {args.timesteps:,} steps. This can take a while...")
    print("Tip: run `tensorboard --logdir logs` in another terminal to watch.\n")

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    model.save(args.save_name)
    print(f"\nDone. Final model saved to {args.save_name}.zip")
    print("Watch it play with:  python src/play.py --model " + args.save_name)


if __name__ == "__main__":
    main()
