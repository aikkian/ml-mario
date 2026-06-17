# Tier 1 Runbook — quick, CPU-friendly wins

Goal: get the 5-stage agent (especially the laggards **1-2** and **1-3**) over
the line using changes that are cheap on your AMD/Windows CPU. Run these on your
PC (commands shown for PowerShell). Watch progress with
`tensorboard --logdir logs`.

> All three experiments below write a different `--save-name`, so they don't
> overwrite each other. Use `--save-freq 250000` on long runs to avoid hundreds
> of checkpoint files.

---

## Experiment 1 — just train longer (baseline)

1-2/1-3 are "almost there," so more steps may be all they need. Continue your
existing run from its latest checkpoint:

```powershell
git checkout claude/smarter-algorithm
git pull
python src\train.py --resume models\mario_ppo_<latest>_steps --levels "1-1,1-2,1-3,4-1,4-4" --eval-levels "1-1,1-2,1-3" --timesteps 9000000 --n-envs 12 --n-epochs 4 --save-freq 250000 --save-name mario_generalist
```
(Find `<latest>` with `Get-ChildItem models\ | Sort-Object LastWriteTime -Descending | Select-Object -First 5`.)

➡️ Watch `eval/flag_rate_1-2` and `eval/flag_rate_1-3` in TensorBoard. Still
climbing = keep going.

---

## Experiment 2 — drop 4-4 (focus capacity on the learnable levels)

The castle maze (4-4) mostly fails and wastes capacity. Train on the other four
so the agent isn't dragged. Start FRESH (different pool, so don't resume the
5-level model):

```powershell
python src\train.py --levels "1-1,1-2,1-3,4-1" --eval-levels "1-1,1-2,1-3" --timesteps 10000000 --n-envs 12 --n-epochs 4 --save-freq 250000 --save-name mario_4lvl
```

➡️ Compare its `eval/flag_rate_1-2` / `1-3` at the same step count vs Experiment
1. If higher, dropping 4-4 helped.

---

## Experiment 3 — reward shaping (denser learning signal)

Adds a bonus for new forward progress + a big flag-completion bonus. Reward scale
differs from unshaped, so **start fresh — do NOT resume an unshaped model**:

```powershell
python src\train.py --levels "1-1,1-2,1-3,4-1" --eval-levels "1-1,1-2,1-3" --shape-reward --timesteps 10000000 --n-envs 12 --n-epochs 4 --save-freq 250000 --save-name mario_shaped
```

➡️ Compare `mario_shaped` vs `mario_4lvl` (both 4-level) at equal steps. Keep
shaping only if its flag rates are clearly higher.

---

## Watching / comparing models

```powershell
# Watch the best-so-far model cycle the levels
python src\play.py --levels "1-1,1-2,1-3,4-1" --model models\mario_4lvl_best --episodes 8

# Inspect a single laggard
python src\play.py --level 1-2 --model models\mario_shaped_best --episodes 5 --fps 30
```

The eval callback saves `<save-name>_best.zip` whenever the overall flag rate
improves — that's the model to watch.

---

## How to decide what stuck

Compare the three runs **at the same step count** on the same eval levels:

| Run | What it tests |
|---|---|
| `mario_generalist` (Exp 1) | does more time alone fix 1-2/1-3? |
| `mario_4lvl` (Exp 2) | does dropping 4-4 help? |
| `mario_shaped` (Exp 3) | does reward shaping help? |

Pick the winner. If 1-2/1-3 still plateau across all three → move to **Tier 2
(curiosity / weighted sampling)** in `SMARTER_TODO.md`.

---

## Tier 1 checklist

- [ ] Exp 1: continue training to ~15M total steps; watch 1-2/1-3 eval curves
- [ ] Exp 2: fresh 4-level run (drop 4-4); compare at equal steps
- [ ] Exp 3: fresh 4-level + `--shape-reward`; compare at equal steps
- [ ] Decide the winner; if all plateau, proceed to Tier 2
