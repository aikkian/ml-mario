# ml-mario 🍄 — Teaching an AI to Play Super Mario Bros by Itself

This project trains a **machine learning agent** to play *Super Mario Bros*
(NES) and beat **World 1-1** entirely on its own — by trying over and over,
millions of times, until it learns how. It's built as a **learning project**:
the code is small, split into one-idea-per-file, and heavily commented.

If you have very little ML background, **start with the "How it works" section
below**, then follow "Quick Start" top to bottom.

---

## How it works (the one idea you need: Reinforcement Learning)

Think about training a dog with treats:

- The dog **tries** something.
- If it's good, it gets a **treat**; if not, nothing (or a "no").
- Over many repetitions, it learns which actions earn treats.

That's exactly what's happening here. The vocabulary:

| Dog analogy | RL term | In this project |
|---|---|---|
| The dog | **Agent** | The thing that learns (a neural network "brain") |
| The world | **Environment** | The Mario game |
| What the dog sees | **Observation** | The game screen (preprocessed) |
| A thing the dog does | **Action** | A button press (run, jump, ...) |
| A treat / scolding | **Reward** | + for moving right & reaching the flag, − for dying / wasting time |
| One training rep | **Episode** | One attempt at the level (start → death or flag) |
| The dog's learned habits | **Policy** | The agent's strategy: "when I see X, press Y" |

**"Keep trying until it passes the level" is literally how this works.** Each
episode is one attempt. The agent plays *millions* of frames; every attempt
nudges its strategy toward more reward. At first it looks random and dumb — that
is normal and expected. Watching it go from random flailing to beating the level
is the whole point.

We use the **PPO** algorithm (a popular, stable RL method) from the
**Stable-Baselines3** library, so you don't have to hand-write the math to get
started.

### Why Python?

Python is the standard language for machine learning. The Mario emulator
binding, the RL algorithms, and almost every tutorial you'll find are all
Python. There's no real competition for this project.

---

## The files (read them in this order)

| File | What it teaches |
|---|---|
| `src/env.py` | **The environment.** How the agent perceives the game (grayscale, resize, frame-skip, frame-stack), stuck-detection, and how we bridge the old/new library APIs. |
| `src/smoke_test.py` | A no-ML sanity check that the game and libraries work. Run this **first**. |
| `src/train.py` | **The training loop.** Creates the PPO agent and runs the learning. |
| `src/callbacks.py` | Auto-saving progress, and detecting the moment Mario beats the level. |
| `src/play.py` | **The payoff.** Load a trained brain and watch (or record) Mario play. |
| `src/_compat.py` | Compatibility shims so the old emulator runs on Python 3.13 / NumPy 2.x. Imported first by every script. |
| `COMMANDS.md` | A copy-paste cheat sheet of every command (setup, train, play, record, replay). |

---

## Quick Start

### 1. Install Python 3.10

The Mario/emulator libraries are happiest on **Python 3.8–3.10** (newer versions
often fail to build). Check yours with `python --version`.

### 2. System requirement: a C compiler

The emulator (`nes-py`) compiles a small bit of C code on install:

- **Linux:** `sudo apt-get install build-essential`
- **macOS:** `xcode-select --install`
- **Windows:** install "Microsoft C++ Build Tools" (Visual Studio Build Tools).

### 3. Create an isolated environment and install dependencies

A *virtual environment* keeps this project's packages separate from the rest of
your computer.

```bash
# from the ml-mario/ folder
python -m venv venv

# activate it:
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows (PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

> Tip: `requirements.txt` already includes `imageio`/`imageio-ffmpeg` for saving
> gameplay videos. If you skipped them, `pip install imageio imageio-ffmpeg`.

### 4. Smoke test — prove the game runs (no ML yet)

```bash
python src/smoke_test.py
```

You should see a Mario window with Mario moving randomly, and a message that the
observation shape is `(4, 84, 84)`. **If this errors, see Troubleshooting.**
Don't move on until this works.

### 5. A quick training test (a few minutes)

Confirm the learning loop runs end-to-end with a tiny number of steps:

```bash
python src/train.py --timesteps 50000
```

It won't be good yet — we're just checking it completes and saves a model.

### 6. Train for real (hours) — in parallel for speed

The emulator runs on the CPU one frame at a time, so the biggest speedup is
running several games at once with `--n-envs` (try a number near your CPU core
count; `sysctl -n hw.physicalcpu` on macOS):

```bash
python src/train.py --timesteps 2000000 --n-envs 8
```

You can stop anytime with **Ctrl+C** and continue later from a checkpoint:

```bash
python src/train.py --resume models/mario_ppo_500000_steps --timesteps 1000000 --n-envs 8
```

In a **second terminal**, watch it learn live:

```bash
tensorboard --logdir logs
```

Open the URL it prints (usually `http://localhost:6006`) and watch the
**episode reward** curve. Trending up = it's learning. Checkpoints are saved to
`models/` along the way, and a special `mario_ppo_FLAG_*` file is saved the
first time it beats the level.

Useful `train.py` flags: `--n-envs`, `--resume`, `--level`, `--device`
(cpu/cuda/mps), `--ent-coef` (exploration), `--learning-rate`, `--save-name`.

### 7. Watch it play

```bash
python src/play.py --model mario_ppo_final --episodes 5
```

Handy `play.py` flags:
- `--stochastic` — play with exploration randomness (often reaches the flag
  earlier in training than the deterministic "best move" default).
- `--fps 20` — slow motion (60 ≈ real time, 0 = full speed).
- `--until-flag` — keep trying and stop on the first win.
- `--record` — save the run as an `.mp4` in `videos/`.
- `--seed N` + `--replay K` — reproduce an exact episode so you can re-watch it.

See `COMMANDS.md` for the full, copy-paste command reference.

---

## What to realistically expect

- **Time:** Beating 1-1 reliably usually takes a **few million steps** —
  roughly a handful of hours on a normal computer (much faster with a GPU,
  slower on CPU-only, but CPU works). The *first* flag-grab often appears
  earlier than "reliable."
- **Early on it looks terrible.** Random flailing for a while is expected.
- **The biggest hurdle is installation**, not the ML. Step 4 exists to catch
  that early.

---

## Experiment (this is where you learn the most)

Change one thing at a time and watch how the reward curve responds. Several
knobs are command-line flags on `train.py`:

- `--learning-rate` — bigger = faster but more unstable.
- `--ent-coef` — exploration strength. **Raise it (e.g. `0.05`) when the agent
  keeps dying at the same spot** (it's stuck in a rut and needs to try new
  things); lower it back to `0.01` once it finds the way, so it sharpens into
  *reliable* wins. This explore-vs-exploit trade-off is a core RL idea.

Knobs still inside the code:
- `gamma`, `n_steps` in `src/train.py`.
- `stuck_steps` (early cut-off for no progress) in `src/env.py`.
- Swap `SIMPLE_MOVEMENT` for `RIGHT_ONLY` in `src/env.py` to shrink the actions.

### Stochastic vs deterministic, and "reliable" wins

During training the agent acts **randomly** (exploring), so a few lucky runs may
reach the flag long before it can do so consistently. `play.py` defaults to
**deterministic** (its single best move) — the true measure of skill. A model
has *mastered* a level when even **deterministic** play reaches the flag
reliably. Use `--stochastic` to watch the exploratory style.

---

## Troubleshooting

Almost every problem here is a **dependency version mismatch** between the old
Mario libraries and the newer RL library.

### Running on Python 3.13 (NumPy 2.0)

The cleanest setup is **Python 3.8–3.10** (with NumPy 1.x). But the project also
runs on **Python 3.13** thanks to `src/_compat.py`, a compatibility shim. On
Python 3.13 you're forced onto NumPy 2.x, whose stricter integer math makes the
old emulator crash with `OverflowError: ... out of bounds for uint8`. The shim
fixes this by **casting the specific overflowing memory reads to plain Python
ints** (in nes-py's ROM loader and gym-super-mario-bros' position code). It also
restores a few NumPy names removed in 2.0.

> Note: NumPy 2.0 briefly offered a global "legacy promotion" switch, but it is
> *ignored* from NumPy 2.2 onward — which is why the shim uses targeted int()
> casts instead of relying on that switch.

Relatedly, `src/env.py` builds the Mario env **directly** (not via
`gym_super_mario_bros.make`) to avoid gym 0.26's auto-wrappers, which assume the
new 5-value step API and crash on the emulator's old 4-value API
("not enough values to unpack (expected 5, got 4)").

To use Python 3.13 you don't need to do anything special — just install and run.
The shim is imported automatically by `smoke_test.py`, `train.py`, `play.py`,
and `env.py`. If you ever import the Mario environment from your own script, add
`import _compat` as the **first** line.

If you'd rather use the rock-solid 3.10 path instead:

```bash
# macOS (Homebrew):
brew install python@3.10
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # numpy<2 is pinned for Python < 3.11
```

**If `pip install -r requirements.txt` or the smoke test fails**, try the older,
very stable combo instead:

```bash
pip install "stable-baselines3==1.8.0" "gym==0.21.0" \
            "gym-super-mario-bros==7.4.0" "nes-py==8.2.1" \
            opencv-python tensorboard
```

> Note: `gym==0.21.0` sometimes needs an older setuptools to install:
> `pip install "setuptools==65.5.0" "wheel<0.40"` first, then the line above.

The code in `src/env.py` is written **defensively** to work with either combo
(it handles both the old 4-value and new 5-value step formats).

**No window appears / display errors on a remote machine:** rendering needs a
display. On a headless server you can still *train* (training uses
`render_mode=None`), just not `play.py`/`smoke_test.py`'s window.

---

## Beyond 1-1

- **Other levels:** pass `--level SuperMarioBros-1-2-v0` (etc.) to both
  `train.py` and `play.py`. Use a separate `--save-name`, and back up `models/`
  first since checkpoint filenames don't include the level.
- **Transfer learning:** start a new level from an already-trained brain with
  `--resume`. It reuses skills like "run right, jump obstacles" and usually
  learns the new level faster than from scratch.
- **Catastrophic forgetting (important):** training a model on 1-2 makes it
  *worse* at 1-1 — RL agents specialize to whatever they last trained on. Keep
  **one model per level**. To get a single model good at *many* levels you must
  train on them *together* (multi-task), not one after another.

## Where to go further

- **Go under the hood:** re-implement the algorithm yourself. The classic
  beginner algorithm to hand-build is **DQN** in PyTorch — a great next project
  once you understand the moving parts you used here.
