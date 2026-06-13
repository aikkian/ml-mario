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

Just want to see it win? Keep trying until it reaches the flag, then stop:
    python src/play.py --model models/mario_ppo_FLAG_386440 --until-flag --stochastic
"""

import _compat  # noqa: F401  (Python 3.13 / NumPy 2.x shims — keep first)

import argparse
import os
import time

import cv2
from stable_baselines3 import PPO

from env import make_mario_env


def write_video(frames, path, fps):
    """Save a list of RGB frames as an .mp4 video."""
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in frames:
        # OpenCV writes BGR; our frames are RGB, so convert.
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def play_episode(env, model, deterministic, step_delay, record=False):
    """Play one full attempt.

    Returns (beat_level, status, total_reward, max_x, frames) where `frames` is a
    list of full-color RGB frames if record=True, else an empty list.
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    beat_level = False
    truncated = False
    max_x = 0          # furthest right Mario reached this attempt
    frames = []

    while not done:
        env.render()
        if step_delay:
            time.sleep(step_delay)  # slow playback down to the target fps
        # deterministic=True -> the agent's single best action (no random
        # exploration). --stochastic flips this on to add exploration.
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated
        max_x = max(max_x, info.get("x_pos", 0))
        if info.get("flag_get"):
            beat_level = True
        if record and env._last_rgb is not None:
            frames.append(env._last_rgb)

    # Explain how the attempt ended: beat the flag, got stuck (our early cut-off
    # truncates with no game-over), or actually died / ran out of time.
    if beat_level:
        status = "🏁 BEAT THE LEVEL!"
    elif truncated:
        status = "stuck (no progress)"
    else:
        status = "died / out of time"
    return beat_level, status, total_reward, max_x, frames


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
    parser.add_argument(
        "--until-flag", action="store_true",
        help="Keep playing attempts and STOP as soon as one reaches the flag. "
             "Best combined with --stochastic. --episodes becomes the max number "
             "of attempts to try before giving up.",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Play with exploration randomness (like during training) instead of "
             "always taking the single best move. Early in training this often "
             "reaches the flag when deterministic play can't yet.",
    )
    parser.add_argument(
        "--fps", type=float, default=60.0,
        help="Playback speed in game frames per second. 60 ~= real time; use a "
             "lower value (e.g. 15 or 30) to watch in slow motion. 0 = as fast "
             "as possible.",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Save a video of each attempt (or, with --until-flag, only the "
             "winning one) into the --video-dir folder. Lets you run fast with "
             "--fps 0 and still re-watch the win afterwards.",
    )
    parser.add_argument(
        "--video-dir", type=str, default="videos",
        help="Folder to save recorded videos into (used with --record).",
    )
    args = parser.parse_args()

    # render_mode="human" opens the game window so you can see it.
    env = make_mario_env(render_mode="human")
    model = PPO.load(args.model)
    deterministic = not args.stochastic

    if args.until_flag and deterministic:
        print("Hint: --until-flag works best with --stochastic — a deterministic "
              "agent plays the same run every time, so if it can't win once it "
              "never will.\n")

    # Each step advances `skip` game frames, so to hit the target fps we pause
    # skip/fps seconds per step. (fps <= 0 means no pausing = full speed.)
    skip = getattr(env, "_skip", 4)
    step_delay = (skip / args.fps) if args.fps > 0 else 0.0
    # Recorded videos: one stored frame = `skip` game frames, so this plays back
    # at roughly real time.
    video_fps = max(round(60 / skip), 1)

    # Both modes try up to `episodes` attempts; --until-flag stops on the first
    # win, plain mode plays them all.
    max_attempts = args.episodes
    won = False
    for episode in range(1, max_attempts + 1):
        beat_level, status, total_reward, max_x, frames = play_episode(
            env, model, deterministic, step_delay, record=args.record
        )
        print(
            f"Episode {episode}: reward={total_reward:.0f}  "
            f"reached x={max_x}  ->  {status}"
        )

        # Save a video of this attempt: every attempt in plain mode, or only the
        # winning attempt in --until-flag mode.
        if args.record and (beat_level or not args.until_flag):
            suffix = "_FLAG" if beat_level else ""
            path = os.path.join(args.video_dir, f"episode_{episode}{suffix}.mp4")
            write_video(frames, path, video_fps)
            print(f"  saved video: {path}")

        if beat_level and args.until_flag:
            won = True
            print(f"\n🎉 Reached the flag on attempt {episode}! Stopping.")
            break

    if args.until_flag and not won:
        print(f"\nNo flag in {max_attempts} attempts. Try more attempts "
              f"(--episodes), add --stochastic, or train the model further.")

    env.close()


if __name__ == "__main__":
    main()
