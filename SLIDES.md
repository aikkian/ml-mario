---
marp: true
theme: default
paginate: true
title: Teaching an AI to Play Super Mario Bros
---

<!--
This is a Marp slide deck. To turn it into PowerPoint / PDF:
  npm install -g @marp-team/marp-cli
  marp SLIDES.md --pptx     # PowerPoint
  marp SLIDES.md --pdf      # PDF
Or paste into any Markdown-slides tool (Marp for VS Code, Slidev, reveal.js).
-->

# 🍄 Teaching an AI to Play Super Mario Bros

### A beginner's reinforcement-learning project

The computer learns to beat the level **on its own** — by trying over and over,
millions of times, until it succeeds.

*No hard-coded moves. No human demonstrations. Just trial, error, and reward.*

---

## The Goal

- Take the classic NES game **Super Mario Bros**
- Let a program (an "**agent**") play it with **zero instructions**
- Reward it for good behavior, and let it **figure out how to win by itself**
- Target: beat **World 1-1**, then **1-2**, and beyond

> "Keep trying until it passes the level" — that is *literally* how it works.

---

## What kind of Machine Learning is this?

**Reinforcement Learning (RL)** — learning by trial and error with rewards.

| Family | How it learns | Example |
|---|---|---|
| Supervised | From labeled examples | "This photo is a cat" |
| Unsupervised | Finds patterns in data | Grouping customers |
| **Reinforcement** ✅ | **Trial + error + rewards** | **Playing a game** |

There is **no dataset of correct moves** — the agent discovers them.

---

## The Big Idea: Training a Dog with Treats 🐕

- The dog **tries** something
- Good action → **treat**; bad action → nothing
- Over many repetitions, it learns what earns treats

**Our "dog" is the AI. The game is its world. The score is its treats.**

---

## The Vocabulary (same idea, real terms)

| Dog analogy | RL term | In this project |
|---|---|---|
| The dog | **Agent** | The neural-network "brain" |
| The world | **Environment** | The Mario game |
| What it sees | **Observation** | The game screen |
| What it does | **Action** | A button press (run, jump…) |
| Treat / scolding | **Reward** | + move right & reach flag / − die, waste time |
| One attempt | **Episode** | One life: start → death or flag |
| Learned habits | **Policy** | "When I see X, press Y" |

---

## Why Python?

- The **standard language** for machine learning
- The Mario emulator, the RL algorithms, and the neural-network tools are all in Python
- Virtually every tutorial and library uses it

➡️ **No real competition for this kind of project.**

---

## The Algorithm: PPO 🧠

We use **PPO — Proximal Policy Optimization**

- A popular, **stable**, beginner-friendly RL algorithm
- From the **Stable-Baselines3** library (so we don't hand-write the math)
- The *"Proximal"* part = it changes its strategy in **small, safe steps** → stable learning

Paired with a **CNN (Convolutional Neural Network)** that reads the screen.

> **PPO learns. A CNN sees.**

---

## How the Agent "Sees" the Game

Raw screen is big and colorful → we simplify it so learning is fast:

1. **Grayscale** — color isn't needed to play
2. **Resize to 84×84** — smaller = faster
3. **Frame-skip (4)** — hold each action a few frames
4. **Frame-stack (4)** — see motion (which way things move)

*An agent is only as smart as what it can perceive.*

---

## The Learning Loop

```
   ┌──────────────────────────────────────────┐
   │   1. Agent looks at the screen            │
   │   2. Picks a button to press (Policy)     │
   │   3. Game responds + gives a Reward       │
   │   4. Agent nudges its brain toward         │
   │      more reward                           │
   └──────────────────────────────────────────┘
          repeat  MILLIONS  of times
```

Random flailing → walking right → jumping enemies → **reaching the flag** 🏁

---

## Project Structure

| File | Role |
|---|---|
| `env.py` | The environment + how the agent perceives it |
| `train.py` | The training loop (PPO) |
| `play.py` | Watch / record a trained agent |
| `callbacks.py` | Save progress + detect the win |
| `_compat.py` | Make old game run on modern Python |
| `smoke_test.py` | Quick "does it work?" check |

*Small files, one idea each, heavily commented — built to learn from.*

---

## Watching It Improve 📈

We track **how far right Mario gets** (`x` position; flag ≈ 3160):

| Training | Result |
|---|---|
| 50k steps | dies at first pipe (x≈310) |
| ~400k steps | reaches **78%** of the level (x≈2475) |
| ~1.2M steps | **reaches the flag** 🏁 |

The agent genuinely gets smarter the more it practices — until it plateaus.

---

## Key Engineering Challenges We Solved

- **Modern Python broke the old game** (NumPy 2.0 integer overflow) → compatibility shim
- **Old vs new library APIs** mismatched → built the environment directly
- **Slow on a laptop** → run **many games in parallel** across CPU cores
- **Wasted time when stuck** → auto-end dead attempts ("stuck detection")
- **Save & replay wins** → record gameplay videos, reproducible runs

---

## Speed: CPU vs GPU

- **The game emulator runs on the CPU** — sequential, one frame at a time
- A GPU **can't** speed up the emulator (it's not parallel math)
- The GPU only helps the **neural-network** part

➡️ **Biggest speedup = many parallel games (CPU cores)**, not a fancy GPU.
A GPU only pays off when paired with *many* parallel environments.

---

## ML Concepts We Learned Along the Way

- **Explore vs. Exploit** — try new things to escape a rut, then sharpen to win reliably
- **Stochastic vs. Deterministic** — lucky random wins come *before* consistent skill
- **Transfer Learning** — reuse a trained brain to learn a new level faster
- **Catastrophic Forgetting** — training on level 2 makes it *worse* at level 1
  → keep **one model per level**

---

## Demo Commands

```bash
# Train (8 games in parallel)
python src/train.py --timesteps 1000000 --n-envs 8

# Watch it play
python src/play.py --model mario_ppo_final --episodes 5

# Keep trying until it wins, and record the video
python src/play.py --model mario_ppo_FLAG --until-flag --stochastic --record
```

---

## Results & Takeaways

- ✅ Built a working AI that **learns Mario from scratch**
- ✅ Beats **1-1** and **1-2**
- ✅ Learned core ML ideas hands-on: RL, PPO, neural networks, tuning

**The journey *was* the lesson:** going from "dies instantly" to "beats the
level" by reward and repetition.

---

## Where To Go Next

- Harder levels & a **generalist** agent (train on many levels at once)
- Tune hyperparameters to learn faster
- **Go under the hood:** build the algorithm (DQN) from scratch in PyTorch

# Thank you! 🍄🎮
