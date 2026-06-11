"""
callbacks.py
============

"Callbacks" are little pieces of code that Stable-Baselines3 runs automatically
at regular points during training. We use two:

1. A CHECKPOINT callback (built into SB3) that saves the agent's "brain" to disk
   every N steps. Training can take hours, so this means a crash or a closed
   laptop won't lose your progress — and you can watch intermediate skill levels.

2. A custom FLAG callback that watches for the moment Mario reaches the
   flagpole (the game reports this via `info["flag_get"] == True`). That is the
   concrete definition of "passed the level." When it happens we print a message
   and save a specially-named model so you can easily find a winning brain.
"""

import os

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


def make_checkpoint_callback(save_dir="models", save_freq=50_000):
    """Save the model every `save_freq` steps into `save_dir`."""
    os.makedirs(save_dir, exist_ok=True)
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix="mario_ppo",
    )


class FlagCallback(BaseCallback):
    """Detects when Mario beats the level (reaches the flag) and saves the brain."""

    def __init__(self, save_dir="models", verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.flags_reached = 0

    def _on_step(self) -> bool:
        # During training the environment(s) report an `infos` list. We scan it
        # for the flag_get signal that means "level complete".
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("flag_get"):
                self.flags_reached += 1
                if self.verbose:
                    print(
                        f"\n🏁  FLAG REACHED! (#{self.flags_reached}) "
                        f"at {self.num_timesteps:,} steps — Mario beat the level!\n"
                    )
                os.makedirs(self.save_dir, exist_ok=True)
                path = os.path.join(
                    self.save_dir, f"mario_ppo_FLAG_{self.num_timesteps}"
                )
                self.model.save(path)
        # Returning True tells SB3 to keep training.
        return True
