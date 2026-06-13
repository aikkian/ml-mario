# ml-mario — Command Cheat Sheet

A quick reference of the commands used in this project. Run them from the
`ml-mario/` folder with your virtual environment **activated**.

---

## 0. One-time setup

```bash
# Create an isolated environment (use Python 3.10 if you hit version issues)
python3 -m venv venv

# Activate it (do this every new terminal session)
source venv/bin/activate            # macOS / Linux
# venv\Scripts\activate             # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Extra packages for saving videos (H.264, plays on macOS/QuickTime)
pip install imageio imageio-ffmpeg
```

Check your Python version and CPU core count:

```bash
python --version                    # want 3.10.x for this stack
sysctl -n hw.ncpu                   # macOS: number of CPU cores (for --n-envs)
```

---

## 1. Sanity check (no machine learning yet)

```bash
# Opens a window, Mario takes random actions — proves the emulator works
python src/smoke_test.py
```

---

## 2. Training

```bash
# Quick test that the training loop runs (a couple of minutes)
python src/train.py --timesteps 50000

# Real training, fast (parallel games across CPU cores)
python src/train.py --timesteps 1000000 --n-envs 8

# Pick a level to train on
python src/train.py --level SuperMarioBros-1-2-v0 --timesteps 1000000 --n-envs 8 --save-name mario_1_2

# Continue (resume) from a saved checkpoint
python src/train.py --resume models/mario_ppo_400000_steps --timesteps 1000000 --n-envs 8

# Use a GPU (on a CUDA/NVIDIA machine; auto-detects by default)
python src/train.py --timesteps 5000000 --n-envs 32 --device cuda
```

**Stop anytime with Ctrl+C** — checkpoints are saved every ~50k frames into
`models/`, and you can resume with `--resume`.

Watch training live (in a **second** terminal, venv activated):

```bash
tensorboard --logdir logs
# then open the printed URL, usually http://localhost:6006
```

---

## 3. Transfer learning (reuse a trained brain on a new level)

```bash
# Keep your old level's checkpoints safe first
mv models models_1_1

# Load the 1-1 brain and keep training it on 1-2
python src/train.py \
  --level SuperMarioBros-1-2-v0 \
  --resume models_1_1/mario_ppo_400000_steps \
  --timesteps 1000000 \
  --n-envs 8 \
  --save-name mario_1_2
```

---

## 4. Watching the agent play

```bash
# Watch the final model (best-move / deterministic play)
python src/play.py --model mario_ppo_final --episodes 5

# Watch a specific checkpoint
python src/play.py --model models/mario_ppo_500000_steps --episodes 5

# Watch a specific level (use the same level the model was trained on)
python src/play.py --level SuperMarioBros-1-2-v0 --model mario_1_2 --episodes 5

# Play with exploration randomness (often reaches the flag earlier in training)
python src/play.py --model models/mario_ppo_FLAG_386440 --episodes 10 --stochastic

# Slow motion (game frames per second): 60 ~= real time, lower = slower, 0 = full speed
python src/play.py --model mario_ppo_final --episodes 5 --fps 20
```

---

## 5. "Just show me a win" + recording

```bash
# Keep trying (fast) until it reaches the flag, then stop
python src/play.py --model models/mario_ppo_FLAG_386440 --until-flag --stochastic --episodes 200 --fps 0

# Same, but also SAVE a video of the winning run into videos/
python src/play.py --model models/mario_ppo_FLAG_386440 --until-flag --stochastic --episodes 200 --fps 0 --record
```

Open a saved video (macOS):

```bash
open videos/episode_36_FLAG.mp4
```

---

## 6. Reproducible runs (replay an exact episode)

```bash
# Search for a flag with a fixed seed so episodes are repeatable
python src/play.py --model models/mario_ppo_FLAG_386440 --until-flag --stochastic --episodes 200 --fps 0 --seed 0
# -> on a win it prints, e.g.:  Replay it anytime with:  --seed 0 --replay 36

# Replay that exact episode at a watchable speed
python src/play.py --model models/mario_ppo_FLAG_386440 --stochastic --seed 0 --replay 36 --fps 30
```

> Note: replay only reproduces a run if you use the **same** `--model`,
> `--level`, `--stochastic`, and `--seed`.

---

## Useful housekeeping

```bash
ls -lt models/              # list checkpoints, newest first
ls -lt videos/              # list recorded videos
git pull                    # get the latest code changes
```
