# Mario AI — Intelligence Assessment & Roadmap to Make It Smarter

## Where we are (observed evidence)

From actual training runs on the user's PC:

- At **~6M steps** (5-level generalist: 1-1, 1-2, 1-3, 4-1, 4-4, PPO + CnnPolicy):
  - ✅ **1-1 solved** (overworld) — wins almost every time
  - ✅ **4-1 solved** (overworld)
  - ⚠️ **1-2 (underground)** — improving, not reliable
  - ⚠️ **1-3 (athletic / pits)** — improving, not reliable
  - ❓ **4-4 (castle maze)** — hardest; likely weak

**Pattern:** the agent is smart at *overworld* levels but struggles with
*structurally different* ones. That's a **capacity + exploration** gap, plus it
simply needs **more training time**.

## Is it "intelligent enough"? — Not yet for the 5-stage goal

It generalizes within a level *style* but can't yet reliably clear the
structurally different stages. There is clear, achievable room to improve. This
document is the plan.

> **Hard constraint:** training runs on an **AMD-on-Windows CPU (~270 fps,
> ~1 hr/million steps)**. PyTorch can't use the AMD GPU (CUDA = NVIDIA only,
> ROCm = Linux only). So every upgrade is judged on **value-per-CPU-second**.
> Heavier methods (LSTM, big CNN, curiosity) cost more compute.

---

## TODO — ordered by value-per-compute

### Tier 1 — cheap / already built (do first)
- [ ] **Train the current run to ~10–15M steps.** 1-2/1-3 are "almost there";
      more steps alone may get them. (Remember: `--n-epochs 4` trades sample
      efficiency for speed, so budget extra steps.)
- [ ] **Test Stage A reward shaping** — start a *fresh* `--shape-reward` run
      (don't resume an unshaped checkpoint — different reward scale). Compare
      `eval/flag_rate_1-2` / `1-3` against the unshaped run **at equal steps**;
      keep shaping only if it helps.
- [ ] **Drop 4-4 from the pool** → `--levels "1-1,1-2,1-3,4-1"`. The maze mostly
      fails and wastes capacity; master the learnable four first, add 4-4 later.

### Tier 2 — the genuine "smarter" upgrades (build next)
- [ ] **Stage B: Curiosity / intrinsic motivation (RND).** Reward exploring
      *novel* states so the agent stops repeating the same fatal move — directly
      targets the 1-2/1-3 "gets stuck" problem. Add `src/curiosity.py` + flags
      `--curiosity rnd --curiosity-coef`. ⚠️ adds a second network → slower on CPU.
- [ ] **Weighted level sampling.** Bias practice toward the laggards (1-2/1-3)
      while still seeing easy levels (prevents catastrophic forgetting). Small
      change to the random-stages selection.

### Tier 3 — bigger levers (need a GPU / more compute)
- [ ] **Stage C: RecurrentPPO (LSTM memory)** via `sb3-contrib`
      (`--algo recurrent_ppo`, `CnnLstmPolicy`). Smarter on timing / moving-
      platform levels. **Slow on CPU.**
- [ ] **Stage D: Bigger IMPALA-style CNN** (`--policy impala`). More capacity to
      represent many different level types. Most compute-hungry.
- [ ] **Cloud / NVIDIA GPU.** Unlocks Tiers C/D at practical speed; the current
      AMD-on-Windows path can't accelerate PyTorch.

### Tier 4 — measure smartness rigorously (ongoing)
- [ ] Use `--eval-levels` with **held-out** stages (e.g. 2-1) and compare
      approaches **at equal step counts** — the only honest way to prove one
      method is smarter, not just different.

---

## Recommended path

1. **Tier 1 first** — more steps + test shaping + drop 4-4. Cheapest, and likely
   enough to handle 1-1 / 1-2 / 1-3 / 4-1.
2. If 1-2/1-3 still plateau → **build Stage B (curiosity)** + **weighted
   sampling**. That's the real "explore more intelligently" lever on CPU.
3. **Tier 3** only once GPU compute is available.

## How we'll know it worked

For each change, compare against the prior run **at the same step count** on the
same eval levels:
- Higher `eval/flag_rate_1-2` / `1-3` = genuinely smarter.
- Track wall-clock too: a method needing 2× compute must clear a meaningfully
  higher bar to be worth it on this machine.

## Honest expectations

- **More steps + reward shaping + curiosity** will most likely get 1-2/1-3 to
  "beats them" — and is the best bet on your CPU.
- **RecurrentPPO / IMPALA** can be smarter but are slower per step; on ~270 fps
  they may lose on wall-clock unless the per-step gain is large.
- **4-4 (maze)** may never be fully reliable — it needs a specific path that's
  genuinely hard for RL.
- There is no silver-bullet algorithm; for *beating more levels*, the combination
  above beats simply "swapping the algorithm."
