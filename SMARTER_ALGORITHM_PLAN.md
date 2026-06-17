# Plan: Smarter Learning for Mario — beyond plain PPO

## Context

The generalist (multi-level PPO) run reached ~6M steps and **solves 1-1 and 4-1
(both *overworld* levels)** but is still weak on **1-2 (underground)** and **1-3
(athletic / lots of pits)**. The pattern suggests two things:
- it generalizes *within* a level style but struggles with **structurally
  different** levels (it needs more *capacity* and better *exploration*), and
- it simply needs **more training** — 1-2/1-3 are "almost there."

Goal: make Mario play **more intelligently** — explore smarter and handle harder,
unfamiliar layouts — not just train longer.

> **Honest framing (read this first).** Plain PPO is already a strong, standard
> choice; there is no drop-in algorithm that trivially beats Mario. Every upgrade
> below **costs more compute**, and this machine is **CPU-bound (~270 fps)** —
> fancier methods will run *slower per step*. So the real question is which
> upgrade buys the most "intelligence per extra second of compute."

---

## The options (ranked by value for *our specific* problem)

### 1. Curiosity / intrinsic motivation — RND or ICM  ⭐ best fit
**What:** add an "intrinsic reward" that pays the agent for reaching *novel*
states, so it explores instead of repeating the same fatal move. RND (Random
Network Distillation) and ICM (Intrinsic Curiosity Module) are the standard
methods.
**Why it fits us:** our struggle (1-2/1-3) is largely an **exploration** problem —
the agent gets stuck because it doesn't try unfamiliar routes. Curiosity directly
targets that and is the most "plays more intelligently" upgrade.
**Cost:** adds a second small network → moderate extra compute; some tuning.
**Effort:** medium (no official SB3 module; add via a reward wrapper / callback).

### 2. Recurrent policy — RecurrentPPO (LSTM memory)
**What:** swap `CnnPolicy` for `CnnLstmPolicy` via **`sb3-contrib`**, giving the
agent short-term **memory** across frames.
**Why:** helps levels where the right action depends on recent history (moving
platforms, off-screen hazards). Closest to a "drop-in smarter algorithm."
**Cost:** LSTM is **noticeably slower on CPU** and needs more steps to train.
**Effort:** low–medium (`pip install sb3-contrib`, switch the model class).

### 3. Bigger / better network — IMPALA-style CNN
**What:** replace SB3's small NatureCNN with a larger residual CNN (custom
`features_extractor`).
**Why:** more capacity to represent *many different* level types at once — directly
addresses "struggles with structurally different levels."
**Cost:** more compute per step (slower on CPU), more memory.
**Effort:** medium (custom features extractor class).

### 4. Reward shaping  (cheap complement, not an algorithm change)
**What:** tweak the reward (e.g. small bonus for new max-x, penalty for
backtracking, bonus for coins/powerups) to give a denser learning signal.
**Why:** often the **highest value-per-effort** change; helps every algorithm.
**Cost:** ~free at runtime. **Effort:** low. Risk: bad shaping can mislead — tune
carefully.

### Also-rans (not recommended now)
- **Rainbow / advanced DQN:** strong but complex, not in core SB3, heavy.
- **DreamerV3 (model-based):** state-of-the-art sample efficiency, but research-
  grade complexity and compute — overkill here.

---

## Recommended path (given CPU limits)

Do the cheap, high-value things first, then one real algorithm upgrade:

1. **Reward shaping** (cheap, helps now) — denser signal for progress on 1-2/1-3.
2. **Curiosity (RND)** — the targeted fix for the exploration problem, the most
   "intelligent" upgrade for our symptom.
3. **(Optional experiment) RecurrentPPO** — try CnnLstm on a couple of levels;
   keep it only if the win justifies the slowdown.
4. **Bigger CNN** — only worth it if you move to a faster machine / GPU later
   (it's the most compute-hungry).

> If raw speed becomes the blocker, the single biggest unlock remains **an NVIDIA
> GPU / cloud GPU** — then the heavier methods (LSTM, IMPALA, curiosity) become
> practical. On the current AMD-on-Windows CPU path, prefer reward shaping +
> light curiosity.

---

## Implementation outline (additive; keeps current PPO working)

### Stage A — Reward shaping (`src/env.py`)
- Add optional shaping in `MarioGymnasium.step` behind a flag, e.g.
  `--shape-reward`: small bonus for beating the episode's max-x, small penalty for
  idling/backtracking, bonus on `flag_get`. Reuse the existing `x_pos` tracking we
  already compute for stuck-detection.

### Stage B — Curiosity (RND) (`src/curiosity.py` + hook in `train.py`)
- A small `RNDModule` (two CNNs: a fixed random target + a trained predictor).
  Intrinsic reward = predictor error (high for novel states). Add it to the
  environment reward via a VecEnv wrapper or a callback. New flags:
  `--curiosity rnd --curiosity-coef 0.1`.

### Stage C — RecurrentPPO option (`src/train.py`)
- `pip install sb3-contrib`; add `--algo {ppo,recurrent_ppo}`. When recurrent,
  use `RecurrentPPO` + `CnnLstmPolicy`. Play.py needs LSTM-state handling in the
  predict loop.

### Stage D — Bigger CNN (`src/policies.py`)
- Custom `BaseFeaturesExtractor` (IMPALA blocks); select via `--policy impala`.

### Docs
- Update README/COMMANDS with the new flags and a short "smarter training" note.

---

## How we'll measure if it's actually smarter

Use the **existing eval callback** (`--eval-levels`) — compare runs *at equal
step counts* on the **same** levels (esp. the laggards 1-2/1-3 and the held-out
2-1):
- Higher `eval/flag_rate_1-2` / `1-3` at the same steps = genuinely smarter.
- Watch **wall-clock too**: a method that needs 2× compute must clear a
  meaningfully higher bar to be worth it on this machine.

---

## Realistic expectations

- **Reward shaping + curiosity** are the most likely to help 1-2/1-3 *and* run on
  your CPU — expect better exploration and faster progress on the hard levels.
- **RecurrentPPO / IMPALA** can be smarter but are **slower per step**; on a
  ~270-fps CPU they may *lose* on wall-clock unless the per-step gain is large.
- None of these is a silver bullet. Honestly, for *beating more levels*, the mix
  of **more steps + reward shaping + curiosity** beats "swap the algorithm."

---

## Open questions before building

1. **Priority:** beat the laggards 1-2/1-3 (→ reward shaping + curiosity) vs.
   experiment with a fundamentally different algorithm (→ RecurrentPPO)?
2. **Compute:** staying on the CPU (favors light methods) or is a GPU/cloud
   option on the table (unlocks LSTM/IMPALA)?
3. **Scope:** start with the cheap Stage A (reward shaping) to see quick gains,
   then add Stage B (curiosity)? Or go straight for RecurrentPPO?
