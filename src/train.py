"""
train.py
========

This is the heart of the project: the TRAINING LOOP. Run this to make Mario
learn.

What happens when you run it:
  1. We build the wrapped Mario environment(s) (from env.py).
  2. We create a PPO agent with a CnnPolicy — a small neural network that looks
     at the screen images and decides which button to press.
  3. `model.learn(...)` runs the loop millions of times: the agent acts, sees
     the reward, and nudges its network to make rewarding actions more likely.
  4. Along the way we save checkpoints and watch for the level being beaten.

SPEED: the emulator runs on the CPU, one frame at a time, so the best way to go
faster is to run several games AT ONCE with --n-envs. On a Mac, try a value near
your number of CPU cores (e.g. 4-8). This multiplies how much experience the
agent gathers per second.

Run it:
    python src/train.py --timesteps 1000000 --n-envs 8   # fast, parallel
    python src/train.py --timesteps 50000                # quick single-env test

Stop anytime with Ctrl+C, then continue from a checkpoint:
    python src/train.py --resume models/mario_ppo_500000_steps --timesteps 500000

Watch progress live (in a second terminal):
    tensorboard --logdir logs
then open the URL it prints (usually http://localhost:6006).
"""

import _compat  # noqa: F401  (Python 3.13 / NumPy 2.x shims — keep first)

import argparse
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import make_mario_env, DEFAULT_LEVEL
from callbacks import make_checkpoint_callback, FlagCallback


def _make_one_env(level):
    """Build a single wrapped Mario env. Monitor records episode reward/length
    for the live charts."""
    env = make_mario_env(level=level, render_mode=None)  # no window = faster
    return Monitor(env)


def build_training_env(n_envs, level):
    """Create `n_envs` Mario environments for the agent to learn from.

    - n_envs == 1: DummyVecEnv (everything in this one process).
    - n_envs > 1:  SubprocVecEnv runs each game in its OWN process, so they
                   genuinely run in parallel across CPU cores. This is the main
                   speed lever on a CPU/Mac.
    """
    env_fns = [partial(_make_one_env, level) for _ in range(n_envs)]
    if n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def main():
    parser = argparse.ArgumentParser(description="Train an RL agent to play Mario.")
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="How many frames to train for. Start small (e.g. 50000) to test, "
             "then go big (1000000+) for real learning.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=8,
        help="How many games to run in parallel. The main speed knob: try a "
             "number near your CPU core count (e.g. 4-8). Use 1 for debugging.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Where the neural network runs: auto / cpu / cuda (NVIDIA GPU) / "
             "mps (Apple GPU). 'auto' picks a GPU if available, else CPU.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a saved model to CONTINUE training from (e.g. "
             "models/mario_ppo_500000_steps). Omit to start fresh.",
    )
    parser.add_argument(
        "--level", type=str, default=DEFAULT_LEVEL,
        help="Which level to train on, e.g. SuperMarioBros-1-1-v0 or "
             "SuperMarioBros-1-2-v0.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="Exploration strength. Higher (e.g. 0.05) makes the agent try more "
             "new things — useful when it keeps dying at the same spot. Lower "
             "makes it more decisive.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="How big each learning step is.",
    )
    parser.add_argument(
        "--save-name", type=str, default="mario_ppo_final",
        help="Filename (without extension) for the final saved model.",
    )
    args = parser.parse_args()

    env = build_training_env(args.n_envs, args.level)
    print(f"Level: {args.level}")

    if args.resume:
        # Load the existing brain and keep training it (don't reset the step
        # counter, so TensorBoard charts continue smoothly).
        print(f"Resuming training from {args.resume} ...")
        model = PPO.load(args.resume, env=env, device=args.device,
                         tensorboard_log="logs")
        # Apply the (possibly new) tuning knobs to the loaded model. Raising
        # --ent-coef here is the usual way to shake a model out of a rut where it
        # keeps dying at the same place.
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        print(f"ent_coef={model.ent_coef}  learning_rate={args.learning_rate}")
        reset_counter = False
    else:
        # Create a fresh PPO agent.
        #   "CnnPolicy" = the brain is a convolutional neural network that reads
        #                 the screen images.
        #   These hyperparameters are sensible starting points for Mario and are
        #   the knobs you'll experiment with later (see the README).
        model = PPO(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            device=args.device,
            tensorboard_log="logs",
            learning_rate=args.learning_rate,  # how big each learning step is
            n_steps=512,          # frames per env collected before each update
            batch_size=64,
            n_epochs=10,
            gamma=0.9,            # how much it values future vs immediate reward
            gae_lambda=1.0,
            ent_coef=args.ent_coef,  # encourages exploration (trying new things)
        )
        reset_counter = True

    # Save a checkpoint roughly every 50k *total* frames. With several parallel
    # envs each callback step covers n_envs frames, so we divide to keep the
    # real interval about the same.
    checkpoint_freq = max(50_000 // args.n_envs, 1)
    callbacks = [
        make_checkpoint_callback(save_dir="models", save_freq=checkpoint_freq),
        FlagCallback(save_dir="models"),
    ]

    print(f"Training for {args.timesteps:,} steps on {args.n_envs} parallel "
          f"env(s), device={model.device}. This can take a while...")
    print("Tip: run `tensorboard --logdir logs` in another terminal to watch.\n")

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=reset_counter)

    model.save(args.save_name)
    print(f"\nDone. Final model saved to {args.save_name}.zip")
    print("Watch it play with:  python src/play.py --model " + args.save_name)


if __name__ == "__main__":
    main()
