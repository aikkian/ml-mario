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
from stable_baselines3.common.utils import set_random_seed

from env import make_mario_env


def write_video(frames, path, fps):
    """Save a list of RGB frames as an .mp4 video.

    Prefers imageio + ffmpeg (H.264), which plays natively on macOS/Windows/web.
    Falls back to OpenCV's mp4v codec — note that mp4v files often appear FROZEN
    in QuickTime/Preview (they play in VLC), which is why H.264 is preferred.
    """
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Preferred: H.264 via imageio-ffmpeg. frames are RGB, which imageio expects.
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    macro_block_size=16)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return
    except Exception as exc:
        print(f"  (imageio H.264 unavailable: {exc}; falling back to OpenCV mp4v)")

    # Fallback: OpenCV mp4v. May not play in QuickTime — try VLC if so.
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in frames:
        # OpenCV writes BGR; our frames are RGB, so convert.
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def play_episode(env, model, deterministic, step_delay, record=False, seed=None):
    """Play one full attempt.

    Returns (beat_level, status, total_reward, max_x, frames) where `frames` is a
    list of full-color RGB frames if record=True, else an empty list.

    If `seed` is given, the random number generators are fixed so the attempt is
    fully reproducible — the same seed always produces the exact same run.
    """
    if seed is not None:
        set_random_seed(seed)
        obs, _ = env.reset(seed=seed)
    else:
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
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Make runs REPRODUCIBLE. With a seed, episode N always plays out "
             "identically, so you can re-watch a specific attempt. The seed used "
             "for each episode is printed so you can replay it later.",
    )
    parser.add_argument(
        "--replay", type=int, default=None,
        help="Replay exactly one episode number from a seeded run (requires "
             "--seed). e.g. --seed 0 --replay 36 re-plays the run that was "
             "episode 36 under seed 0.",
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

    def episode_seed(episode):
        """The seed for a given episode number (None if --seed wasn't given)."""
        return None if args.seed is None else args.seed + episode - 1

    def save_if_recording(episode, beat_level, frames):
        if args.record and (beat_level or not args.until_flag):
            suffix = "_FLAG" if beat_level else ""
            path = os.path.join(args.video_dir, f"episode_{episode}{suffix}.mp4")
            write_video(frames, path, video_fps)
            print(f"  saved video: {path}")

    # --replay N: reproduce exactly one episode from a seeded run, then exit.
    if args.replay is not None:
        if args.seed is None:
            print("--replay needs --seed (the seed the original run used).")
            env.close()
            return
        episode = args.replay
        seed = episode_seed(episode)
        beat_level, status, total_reward, max_x, frames = play_episode(
            env, model, deterministic, step_delay, record=args.record, seed=seed
        )
        print(f"Replay of episode {episode} (seed={seed}): "
              f"reward={total_reward:.0f}  reached x={max_x}  ->  {status}")
        save_if_recording(episode, beat_level, frames)
        env.close()
        return

    # Both modes try up to `episodes` attempts; --until-flag stops on the first
    # win, plain mode plays them all.
    max_attempts = args.episodes
    won = False
    for episode in range(1, max_attempts + 1):
        seed = episode_seed(episode)
        beat_level, status, total_reward, max_x, frames = play_episode(
            env, model, deterministic, step_delay, record=args.record, seed=seed
        )
        seed_note = f" (seed={seed})" if seed is not None else ""
        print(
            f"Episode {episode}{seed_note}: reward={total_reward:.0f}  "
            f"reached x={max_x}  ->  {status}"
        )

        save_if_recording(episode, beat_level, frames)

        if beat_level and args.until_flag:
            won = True
            print(f"\n🎉 Reached the flag on attempt {episode}! Stopping.")
            if args.seed is not None:
                print(f"Replay it anytime with:  --seed {args.seed} "
                      f"--replay {episode}")
            break

    if args.until_flag and not won:
        print(f"\nNo flag in {max_attempts} attempts. Try more attempts "
              f"(--episodes), add --stochastic, or train the model further.")

    env.close()


if __name__ == "__main__":
    main()
