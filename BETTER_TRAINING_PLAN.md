# Plan: A "Generalist" Mario Agent That Plays the Game By Itself

## Context — why this plan exists

The current project trains **one model per level**. It works (we beat 1-1 and
1-2), but it has a fundamental limit we hit head-on:

- A model trained on 1-2 became **worse at 1-1** — *catastrophic forgetting*.
- Each model essentially **memorizes one level's layout** rather than learning
  general skills ("avoid pits, stomp enemies, time jumps").

Goal of this plan: train a **single agent that can play many levels on its own**
— a *generalist* — instead of a separate specialist per level. This is what
"Mario plays the game by itself" really means: drop it into a level it wasn't
specifically tuned on and have it cope.

> **Scope note:** this plan targets a *generalist across several levels*. If the
> real goal is just "beat one level more reliably," that's a smaller effort
> (more training + tuning) — say the word and we'll scope down.

---

## First: can we reuse the 1-1 and 1-2 models?

Short answer: **partially — as a head start, not as a merge.**

- ❌ **You cannot "combine" the 1-1 model and the 1-2 model into one.** Neural
  networks don't merge like files; there's no way to fuse two sets of weights
  into a model that's good at both.
- ✅ **You CAN warm-start** the new generalist training from one existing model
  (transfer learning via `--resume`). The 1-1 brain already knows "run right,
  jump obstacles," which is a useful starting point.
- ⚠️ But because of catastrophic forgetting, the *right* way to get a model good
  at many levels is to **train it on those levels together** (mixed), not to
  stack one after another. Warm-starting just makes that training converge a bit
  faster.

**Decision:** start the generalist run *optionally* warm-started from the best
existing 1-1 model, but train on a **mix of levels simultaneously**.

---

## The core idea: train on many levels at once (domain randomization)

Instead of every parallel environment playing the same level, each one plays a
**randomly chosen level** (and re-randomizes every episode). The agent can no
longer memorize a single layout — to score well it must learn skills that
**transfer across levels**. This is the single most important change.

`gym-super-mario-bros` already supports this via its **random-stages**
environment (`SuperMarioBrosRandomStagesEnv` / the `SuperMarioBrosRandomStages-v0`
id), which picks a random stage from a provided list on each reset. We'll build
on that. *(Confirm the exact class/import during implementation, since the
container here has no emulator installed.)*

---

## Supporting improvements (ranked by expected impact)

1. **Multi-level training** *(the big one — above).* One model, many levels.
2. **Curriculum learning.** Start with easier levels (1-1, 1-2), then expand the
   pool to harder ones (1-3, 1-4, 2-1, …) once success rate is decent. Learning
   hard levels cold is very slow; ramping difficulty is much faster.
3. **Reward normalization (`VecNormalize`).** Different levels give different
   reward scales; normalizing stabilizes learning across them.
4. **A proper evaluation callback.** Periodically test the agent on a **held-out
   set of levels it doesn't train on**, and log the **flag-reach success rate**.
   This is how we *measure generalization* (not just training reward).
5. **Bigger / better policy network.** The default SB3 NatureCNN is small. An
   IMPALA-style CNN (or a wider net) has more capacity for many levels. Optional,
   costs compute.
6. **Tuned PPO hyperparameters.** Larger rollout (`n_steps`), more envs, a
   **learning-rate schedule** (decay over time), and an **entropy schedule**
   (more exploration early, less later). Current values (`n_steps=512`,
   `gamma=0.9`) are tuned for one easy level.
7. **Optional: recurrent policy (LSTM).** `RecurrentPPO` from `sb3-contrib` adds
   memory, which can help on levels needing it. Higher complexity — treat as a
   later experiment, not a first step.

---

## Implementation outline (what we'd actually build)

All additive — keeps the existing single-level flow working.

### `src/env.py`
- Add `make_multi_level_env(levels, render_mode, stuck_steps)` that wraps the
  **random-stages** env (chooses from `levels` each reset). Keep the existing
  `make_mario_env` for single-level use.
- Reuse the existing `MarioGymnasium` preprocessing + stuck-detection wrapper
  unchanged (it already handles the old/new API and the `_compat` shims).

### `src/train.py`
- New flags:
  - `--levels "1-1,1-2,1-3,1-4"` — pool of stages to train on (mutually
    exclusive with the single `--level`).
  - `--curriculum` — start with the first N levels, widen the pool as the
    success rate crosses a threshold.
  - `--normalize` — wrap envs in `VecNormalize` (reward normalization).
  - `--policy {cnn,impala}` and LR/entropy **schedule** options.
- Build the vec env from the level pool; keep `--n-envs`, `--resume`,
  `--device`, etc.

### `src/callbacks.py`
- Add `EvalSuccessCallback`: every K steps, run the agent on a **held-out level
  list**, log mean reward + **flag-reach rate** to TensorBoard, and save a
  `mario_generalist_best` when it improves.

### `src/play.py`
- Add `--levels` so you can watch the generalist cycle through several stages.

### Docs
- Update `README.md` / `COMMANDS.md` with the generalist workflow; note that
  generalist models live under a distinct `--save-name` (e.g.
  `mario_generalist`) so they never clash with the per-level specialists.

---

## How we'll measure success

The metric is **generalization**, not raw training reward:

- **Training success rate:** % of episodes reaching the flag across the training
  level pool (logged by the eval callback).
- **Held-out success rate:** % of flags on levels the agent **never trained on**
  (e.g. train on 1-1/1-2/1-3, test on 1-4). This is the real test of "plays by
  itself."
- Target for a first milestone: reliably beat the **training** levels, and clear
  a meaningful fraction of **held-out** levels.

---

## Verification (run on the user's machine — no emulator in this container)

1. `python -m py_compile src/*.py` after edits (works here).
2. Smoke: `python src/train.py --levels "1-1,1-2" --timesteps 50000 --n-envs 8`
   completes and saves a model.
3. Watch: `python src/play.py --levels "1-1,1-2" --model mario_generalist`
   cycles through both levels.
4. TensorBoard (`tensorboard --logdir logs`): the **eval success-rate** curve
   trends up.
5. Generalization check: evaluate on a **held-out** level and confirm a non-zero
   flag rate.

---

## Realistic expectations

- A generalist needs **substantially more training** than a single-level
  specialist — it's solving a harder problem. Plan for more steps / longer runs
  (parallel envs + patience), and use the curriculum to speed it up.
- Early on it may look *worse* than a specialist on any single level; the payoff
  is that one model handles *many* levels and copes with unfamiliar ones.
- This is genuinely closer to "real" deep-RL research (e.g. how agents are
  trained on Atari/Procgen suites) — a great next learning step.

---

## Open questions to confirm before building

1. **Goal:** generalist across many levels (this plan) **vs.** just beat one
   level more reliably (smaller effort)?
2. **Level pool:** which stages to start with (default: 1-1, 1-2, 1-3, 1-4)?
3. **Warm-start:** initialize from the existing 1-1 model, or train fresh?
4. **Compute:** stay on the Mac (more parallel envs, longer runs) or move to a
   GPU box (then `--device cuda` + many envs)?
