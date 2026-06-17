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


class SuccessEvalCallback(BaseCallback):
    """Periodically MEASURES how good the agent actually is.

    Every `eval_freq` steps it plays a few deterministic episodes on each level in
    `eval_levels` and records, per level and overall:
      - flag_rate: the fraction of attempts that REACH THE FLAG (true skill),
      - mean_reward.
    Include levels the agent does NOT train on (e.g. 2-1) to measure
    GENERALIZATION. Results are logged to TensorBoard (the `eval/` charts) and the
    best-so-far model (by overall flag rate) is saved.

    This is how you tell whether a generalist is really learning — training reward
    alone doesn't show whether it can finish a level it's never seen.
    """

    def __init__(self, eval_levels, eval_freq=250_000, n_eval_episodes=3,
                 save_dir="models", save_name="mario_best", verbose=1):
        super().__init__(verbose)
        self.eval_levels = list(eval_levels)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        self.save_name = save_name
        self._next_eval = eval_freq
        self._envs = {}            # one reusable env per level
        self.best_flag_rate = -1.0

    def _get_env(self, level):
        if level not in self._envs:
            # Imported here (not at top) so this module stays import-light.
            from env import make_mario_env
            self._envs[level] = make_mario_env(level=level, render_mode=None)
        return self._envs[level]

    def _eval_one_level(self, level):
        """Return (flag_rate, mean_reward) over n_eval_episodes on this level."""
        env = self._get_env(level)
        flags = 0
        total_rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            reached_flag = False
            while not done:
                # deterministic = the agent's true best play (no luck).
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += reward
                done = terminated or truncated
                if info.get("flag_get"):
                    reached_flag = True
            flags += int(reached_flag)
            total_rewards.append(ep_reward)
        flag_rate = flags / self.n_eval_episodes
        mean_reward = sum(total_rewards) / len(total_rewards)
        return flag_rate, mean_reward

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.eval_freq

        if self.verbose:
            print(f"\n[eval @ {self.num_timesteps:,} steps] "
                  f"({self.n_eval_episodes} episodes/level)")
        rates = []
        for level in self.eval_levels:
            flag_rate, mean_reward = self._eval_one_level(level)
            rates.append(flag_rate)
            self.logger.record(f"eval/flag_rate_{level}", flag_rate)
            self.logger.record(f"eval/mean_reward_{level}", mean_reward)
            if self.verbose:
                print(f"  {level:>22}: flag {flag_rate*100:5.0f}%   "
                      f"mean_reward {mean_reward:7.0f}")

        overall = sum(rates) / len(rates)
        self.logger.record("eval/flag_rate_overall", overall)

        # Save the best generalist so far (by overall flag rate).
        if overall > self.best_flag_rate:
            self.best_flag_rate = overall
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"{self.save_name}_best")
            self.model.save(path)
            if self.verbose:
                print(f"  ✓ new best overall flag rate "
                      f"{overall*100:.0f}%  ->  {path}.zip")
        return True
