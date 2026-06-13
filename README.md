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
| `src/env.py` | **The environment.** How the agent perceives the game (grayscale, resize, frame-skip, frame-stack) and how we bridge the old/new library APIs. |
| `src/smoke_test.py` | A no-ML sanity check that the game and libraries work. Run this **first**. |
| `src/train.py` | **The training loop.** Creates the PPO agent and runs the learning. |
| `src/callbacks.py` | Auto-saving progress, and detecting the moment Mario beats the level. |
| `src/play.py` | **The payoff.** Load a trained brain and watch Mario play in a window. |

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

### 6. Train for real (hours)

```bash
python src/train.py --timesteps 2000000
```

In a **second terminal**, watch it learn live:

```bash
tensorboard --logdir logs
```

Open the URL it prints (usually `http://localhost:6006`) and watch the
**episode reward** curve. Trending up = it's learning. Checkpoints are saved to
`models/` along the way, and a special `mario_ppo_FLAG_*` file is saved the
first time it beats the level.

### 7. Watch it play

```bash
python src/play.py --model mario_ppo_final
# or an earlier checkpoint to compare skill levels:
python src/play.py --model models/mario_ppo_500000_steps
```

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

Once it's training, try changing one thing at a time in `src/train.py` and watch
how the reward curve responds. Good knobs to explore:

- `learning_rate` — bigger = faster but more unstable.
- `gamma` — how much the agent values future vs. immediate reward.
- `ent_coef` — higher = more exploration (trying new things).
- `n_steps` — how much experience it gathers before each update.
- In `src/env.py`, swap `SIMPLE_MOVEMENT` for `RIGHT_ONLY` (from
  `gym_super_mario_bros.actions`) to shrink the action set even further.

---

## Troubleshooting

Almost every problem here is a **dependency version mismatch** between the old
Mario libraries and the newer RL library.

**`OverflowError: Python integer 1024 out of bounds for uint8` (or a NumPy 2.0
warning):** You're on NumPy 2.0 and/or Python 3.11+. The old emulator only works
with **NumPy 1.x** and **Python 3.8–3.10**. NumPy 1.x has no build for Python
3.13, so the fix is to recreate your virtual environment with **Python 3.10**:

```bash
# macOS (Homebrew):
brew install python@3.10
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # numpy<2 is pinned here
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

## Where to go next

- Beat a harder level — change `DEFAULT_LEVEL` in `src/env.py` (e.g.
  `SuperMarioBros-1-2-v0`).
- Run several game copies in parallel to train faster (vectorized envs).
- **Go under the hood:** re-implement the algorithm yourself. The classic
  beginner algorithm to hand-build is **DQN** in PyTorch — a great next project
  once you understand the moving parts you used here.
