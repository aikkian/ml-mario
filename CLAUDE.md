# CLAUDE.md

Guidance for Claude (and humans) working in this repo.

## What this is

A beginner-oriented **reinforcement learning** project: train an agent to play
*Super Mario Bros* (NES) with **PPO** (Stable-Baselines3 + a CNN policy). Built
as a learning project — small files, one idea each, heavily commented. The user
is new to ML, so favor clear explanations over jargon.

## Layout

- `src/env.py` — `MarioGymnasium`: gymnasium adapter over the old-gym Mario env,
  observation preprocessing (grayscale, resize 84×84, frame-skip 4, frame-stack
  4), and **stuck-detection** (truncate after no rightward progress).
  `make_mario_env(level, render_mode, stuck_steps)` is the factory.
- `src/train.py` — PPO training. Flags: `--timesteps --n-envs --resume --device
  --level --ent-coef --learning-rate --save-name`.
- `src/play.py` — watch/record a model. Flags: `--model --level --episodes
  --stochastic --fps --until-flag --record --video-dir --seed --replay`.
- `src/callbacks.py` — checkpointing + `FlagCallback` (saves `mario_ppo_FLAG_*`
  when `info["flag_get"]`).
- `src/_compat.py` — Python 3.13 / NumPy 2.x shims. **Imported first** by every
  entry script and by `env.py`.
- `COMMANDS.md` — copy-paste command cheat sheet. `README.md` — the tutorial.

## Critical gotchas (these were hard-won)

1. **`train.py` vs `play.py`.** The user repeatedly confuses these. `train.py`
   improves a model (`--resume/--timesteps/--n-envs/--save-name`); `play.py`
   only watches (`--model`). If a command has training flags, it's `train.py`.
2. **NumPy 2.x overflow.** On Python 3.13 the emulator hits
   `OverflowError: ... out of bounds for uint8`. Fixed in `_compat.py` by
   casting the overflowing reads to `int` (nes-py `_rom.py`; smb_env
   `_x_position`/`_left_x_position`/`_y_position`). Do **not** rely on
   `NPY_PROMOTION_STATE=legacy` — ignored since NumPy 2.2.
3. **gym 0.26 API mismatch.** `env.py` builds `SuperMarioBrosEnv` directly to
   bypass gym's TimeLimit/OrderEnforcing wrappers (they expect the 5-tuple step
   API; nes-py returns 4). `MarioGymnasium.step` handles both tuple shapes.
4. **Per-level models / catastrophic forgetting.** Training on a new level
   erodes skill on the old one. One model per level; checkpoint filenames don't
   encode the level, so back up `models/` before training a new one.
5. **Videos:** record as H.264 via imageio/imageio-ffmpeg (OpenCV `mp4v`
   appears frozen in QuickTime).
6. **`--n-envs`** uses SubprocVecEnv (separate processes) — needs the
   `if __name__ == "__main__"` guard, which is present. It's the main CPU
   speedup; the GPU only helps with many parallel envs.

## Dev workflow

- Develop on branch **`claude/nice-edison-w7h74f`**; commit and push there
  (`git push -u origin claude/nice-edison-w7h74f`). Don't push elsewhere without
  asking. Don't open PRs unless asked.
- After editing a `.py`, sanity-check with `python -m py_compile src/<file>.py`
  (the container has no emulator/GPU/display, so we can't run training here).
- Keep README/COMMANDS.md in sync when adding flags or changing behavior.
- `models/`, `logs/`, `videos/` are git-ignored.

## Environment

Targets macOS, Python 3.13, NumPy 2.x. Stable-Baselines3 2.x (gymnasium API);
gym-super-mario-bros 7.4 / nes-py 8.2 (legacy gym). Versions pinned in
`requirements.txt` (`numpy<2` only for Python < 3.11).
