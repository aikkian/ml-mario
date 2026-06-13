"""
_compat.py
==========

Compatibility shims that let the (older) Mario / nes-py / gym stack run on
**Python 3.13 + NumPy 2.x**.

Why this file exists:
  Python 3.13 has no NumPy 1.x build, so you're forced onto NumPy 2.0. NumPy 2.0
  changed how integer math works ("NEP 50"): an expression like
  `some_uint8_value * 1024` used to be quietly widened to a bigger integer type,
  but NumPy 2.0 now raises `OverflowError: ... out of bounds for uint8`. The
  emulator (nes-py) and the Mario reward code do exactly this kind of arithmetic
  when reading the game's memory, so they crash.

The fix:
  Switch NumPy back to its old ("legacy") value-based integer promotion *before*
  any emulator code runs. That one change covers all the overflow sites at once.
  We also restore a few NumPy attribute names that were removed in 2.0 and that
  old libraries may still reference.

IMPORTANT: import this module FIRST — before numpy, gym, nes_py, or
stable_baselines3 — so the settings take effect early. The entry-point scripts
(smoke_test.py, train.py, play.py) and env.py all do `import _compat` as their
very first import.
"""

import os

# The most reliable switch is the environment variable, which NumPy reads when
# it is first imported. setdefault so we never override a user's explicit value.
os.environ.setdefault("NPY_PROMOTION_STATE", "legacy")

import numpy as np  # noqa: E402  (must come after the env var above)

# Belt-and-suspenders: also flip it at runtime, in case NumPy was already
# imported by something else before this module loaded. Wrapped in try/except
# because this private API may not exist on every NumPy version.
try:
    np._set_promotion_state("legacy")
except (AttributeError, TypeError, ValueError):
    pass

# Restore NumPy names that were removed in 2.0 but that old gym / nes-py code may
# still reference. Harmless if they already exist.
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

# Extra safety for the specific ROM-size overflow in nes-py: force the header
# byte values to plain Python ints so the `* 1024` can never overflow, even if
# the legacy promotion switch above is unavailable on this NumPy version.
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
    # nes-py not installed yet, or its internals changed — the legacy promotion
    # switch above is the primary fix regardless.
    pass
