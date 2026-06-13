"""
_compat.py
==========

Compatibility shims that let the (older) Mario / nes-py / gym stack run on
**Python 3.13 + NumPy 2.x**.

Why this file exists:
  Python 3.13 has no NumPy 1.x build, so you're forced onto NumPy 2.0+. NumPy
  2.0 changed integer math ("NEP 50"): an expression like `some_uint8 * 256`
  used to be quietly widened, but now raises
  `OverflowError: Python integer 256 out of bounds for uint8`. The emulator
  (nes-py) and the Mario game code do exactly this when reading the game's
  memory, so they crash.

  NumPy 2.0 briefly offered a global "legacy promotion" switch to restore the
  old behavior, but it is IGNORED from NumPy 2.2 onward — so we cannot rely on
  it. Instead we patch the handful of specific spots that overflow, casting the
  memory reads to plain Python ints (which never overflow). This is independent
  of the NumPy version.

The patches:
  1. Restore a few NumPy attribute names removed in 2.0 (harmless if present).
  2. nes-py ROM header sizes -> int (fixes the ROM-loading overflow).
  3. gym-super-mario-bros position properties -> int (fixes the in-game
     overflow seen in `_x_position`, `_left_x_position`, `_y_position`).

IMPORTANT: import this module FIRST — before numpy, gym, nes_py, or
stable_baselines3. The entry-point scripts (smoke_test.py, train.py, play.py)
and env.py all do `import _compat` as their very first import.
"""

import numpy as np


# 1) Restore NumPy names removed in 2.0 that old gym / nes-py code may reference.
_REMOVED_ALIASES = {
    "bool8": np.bool_,
    "float_": np.float64,
    "int_": np.int64,
    "complex_": np.complex128,
    "unicode_": np.str_,
    "object0": np.object_,
}
for _name, _replacement in _REMOVED_ALIASES.items():
    if not hasattr(np, _name):
        try:
            setattr(np, _name, _replacement)
        except Exception:
            pass


# 2) nes-py: force the ROM header byte values to plain ints so `* 1024` is safe.
try:
    import nes_py._rom as _rom

    def _as_int_property(prop_name):
        original = getattr(_rom.ROM, prop_name).fget

        def getter(self, _orig=original):
            return int(_orig(self))

        setattr(_rom.ROM, prop_name, property(getter))

    for _prop in ("prg_rom_size", "chr_rom_size"):
        try:
            _as_int_property(_prop)
        except Exception:
            pass
except Exception:
    pass


# 3) gym-super-mario-bros: the game reads RAM bytes and does arithmetic that
#    overflows uint8 under NumPy 2.x. Re-define the offending position
#    properties to cast each RAM read to a plain Python int first. These RAM
#    addresses are stable across the 7.x line of gym-super-mario-bros.
try:
    import gym_super_mario_bros.smb_env as _smb

    _Env = _smb.SuperMarioBrosEnv

    def _x_position(self):
        # page (0x6d) * 256 + fine-x (0x86)
        return int(self.ram[0x6D]) * 0x100 + int(self.ram[0x86])

    def _left_x_position(self):
        # pixels from the left edge of the screen
        return (int(self.ram[0x86]) - int(self.ram[0x071C])) % 256

    def _y_position(self):
        # if Mario is above the viewport, add a full screen height
        if self._y_viewport < 1:
            return int(self._y_pixel) + 256
        return int(self._y_pixel)

    _Env._x_position = property(_x_position)
    _Env._left_x_position = property(_left_x_position)
    _Env._y_position = property(_y_position)
except Exception:
    # If gym-super-mario-bros isn't importable yet or its internals changed,
    # skip — env.py will still try, and any remaining overflow will surface
    # with a clear traceback we can patch the same way.
    pass
