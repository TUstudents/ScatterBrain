# Plan: Unified Logging & Standardized Error Messages

## Current State (Audit Summary)

| File | print() (prod) | warnings.warn() | Exceptions raised | Returns None on fail | Custom exceptions |
|---|---|---|---|---|---|
| utils.py | 0 | 0 | 2 | — | 4 defined, **never raised** |
| core.py | **1** | 0 | 12 | — | — |
| io.py | 0 | 6 | 9 | — | — |
| guinier.py | **1** | 9 | 1 | 6 | — |
| porod.py | 0 | 6 | 1 | 5 | — |
| fitting.py | 0 | 9 | 4 | 3 | — |
| visualization.py | 0 | 2 | 3 | — | — |

**Key problems:**
1. Two production `print()` calls that go to stdout unconditionally
2. 32 `warnings.warn()` calls with no logging infrastructure
3. Four custom exceptions (`AnalysisError`, `FittingError`, `ProcessingError`, `ScatterBrainError`) defined but **never raised**
4. Analysis/fitting functions return `None` on failure while core raises exceptions — inconsistent contract
5. Typo in `guinier.py:158` warning message (`"non-negative slope . "`)
6. No logger configured in `__init__.py` — library emits noise by default

---

## Goals

1. Replace all production `print()` with `logging` calls at appropriate levels
2. Wire up a standard `logging.getLogger("scatterbrain")` hierarchy (NullHandler by default — library best practice)
3. Wire up `warnings.warn()` calls into the logging system via `logging.captureWarnings(False)` + direct `logger.warning()` — keeps the API but routes output through the logging framework
4. Wire the defined but unused custom exceptions into the right raise sites
5. Standardize failure signaling in analysis/fitting: keep `return None` for *expected* soft failures (not enough data), convert *programming/API misuse* failures to exceptions — document the convention clearly
6. Fix the typo and other minor message quality issues
7. Add a `scatterbrain.configure_logging()` convenience helper so users can turn on debug output in one call

---

## Implementation Steps

### Step 1 — Create the logger in `utils.py` and configure `__init__.py`

**`scatterbrain/utils.py`**
- Add at module top (after imports):
  ```python
  import logging
  logging.getLogger("scatterbrain").addHandler(logging.NullHandler())
  ```
- This is the single canonical place the NullHandler is registered (PEP 3148 / logging HOWTO best practice).

**`scatterbrain/__init__.py`**
- Add a `configure_logging(level=logging.DEBUG, handler=None)` convenience function:
  ```python
  def configure_logging(level=logging.DEBUG, handler=None):
      """Enable scatterbrain log output. Call once at application startup."""
      logger = logging.getLogger("scatterbrain")
      if handler is None:
          handler = logging.StreamHandler()
          handler.setFormatter(logging.Formatter(
              "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
          ))
      logger.addHandler(handler)
      logger.setLevel(level)
  ```

### Step 2 — Replace production `print()` calls

**`scatterbrain/core.py:296`**
- Replace: `print("Warning: Metadata could not be deep-copied...")`
- With: `logging.getLogger("scatterbrain.core").warning("Metadata could not be deep-copied during to_dict; using shallow copy.")`

**`scatterbrain/analysis/guinier.py:151`**
- Replace: `print(f"Initial slope: {slope_initial:.3g} ...")`
- With: `logger.debug("Auto-range initial slope: %.3g (expected negative for Guinier fit)", slope_initial)`
- Note: This is a debug diagnostic — DEBUG level is correct.

### Step 3 — Add per-module loggers

Add to the top of each module (after imports):

```python
import logging
logger = logging.getLogger(__name__)
```

Modules to update: `core.py`, `io.py`, `analysis/guinier.py`, `analysis/porod.py`, `modeling/fitting.py`, `modeling/form_factors.py`, `visualization.py`

This produces the hierarchy:
```
scatterbrain                    (NullHandler set in utils.py)
├── scatterbrain.core
├── scatterbrain.io
├── scatterbrain.analysis.guinier
├── scatterbrain.analysis.porod
├── scatterbrain.modeling.fitting
└── scatterbrain.visualization
```

### Step 4 — Convert `warnings.warn()` to `logger.warning()`

Replace every `warnings.warn(msg, UserWarning)` with `logger.warning(msg)` in:
- `io.py` (6 calls)
- `analysis/guinier.py` (9 calls — except the ones that also return None, keep same logic)
- `analysis/porod.py` (6 calls)
- `modeling/fitting.py` (9 calls)
- `visualization.py` (2 calls)

**Logging level mapping:**

| Current pattern | New level | Rationale |
|---|---|---|
| Data parsing issues (non-numeric, dropped rows) | `WARNING` | Caller may not expect data loss |
| Fit failed (not enough points, bad q-range) | `WARNING` | Soft failure, function returns None |
| Fit parameter diagnostic (slope, Rg estimate) | `DEBUG` | Internal calculation detail |
| `OptimizeWarning` from scipy | `WARNING` | Third-party warning |
| Metadata function errors | `WARNING` | Unexpected but non-fatal |
| Debug diagnostics (guinier auto-range) | `DEBUG` | Internal only |

Remove `import warnings` from any file where it is no longer needed after the conversion.

### Step 5 — Wire in the custom exceptions

Currently defined in `utils.py` but never raised. Map them to raise sites:

| Exception | Where to raise | Replace |
|---|---|---|
| `AnalysisError` | `guinier.py` / `porod.py` — when input `curve` is wrong type or invalid | Currently raises bare `TypeError` |
| `FittingError` | `fitting.py` — when `initial_params` length wrong, bounds wrong | Currently raises bare `ValueError` |
| `ProcessingError` | `core.py` — unit conversion failure at line 394 | Currently raises bare `ValueError` |
| `ScatterBrainError` | Keep as base only; do not raise directly | — |

**Exact substitutions:**

`guinier.py:95`
```python
# Before
raise TypeError("Input 'curve' must be a ScatteringCurve1D object.")
# After
raise AnalysisError("Input 'curve' must be a ScatteringCurve1D object.")
```

`porod.py:95`
```python
raise AnalysisError("Input 'curve' must be a ScatteringCurve1D object.")
```

`fitting.py:94`
```python
raise FittingError("Input 'curve' must be a ScatteringCurve1D object.")
```

`fitting.py:151-154` (param length mismatch)
```python
raise FittingError(f"Length of initial_params ({len(initial_params)}) does not match ...")
```

`fitting.py:159-162` (bounds length mismatch)
```python
raise FittingError(f"Length of param_bounds components must match ...")
```

`core.py:394` (unit conversion failure)
```python
raise ProcessingError(f"Failed to convert q units: {e}") from e
```

Add imports at top of each affected file:
```python
from scatterbrain.utils import AnalysisError, FittingError, ProcessingError
```

### Step 6 — Standardize the `None`-return convention and document it

**Decision:** Keep `return None` for **soft, data-driven failures** (insufficient points, invalid q-range). These are expected conditions during exploratory data analysis — raising an exception would be disruptive.

**Add a docstring convention note** to each analysis function:

```
Returns
-------
dict or None
    Result dictionary on success. Returns None if the fit could not be
    performed (e.g., insufficient data points); a WARNING-level log message
    is emitted describing the reason.
```

This is the only documentation change required — no behavioral change for soft failures.

### Step 7 — Fix message quality issues

| Location | Current message | Fixed message |
|---|---|---|
| `guinier.py:158` | `"Guinier fit (auto-range): Initial fit yielded non-negative slope . "` | `"Guinier fit (auto-range): Initial fit yielded non-negative slope; Guinier approximation may not be valid in this q-range."` |
| `fitting.py:121` | `"...Using absolute errors, or ignoring errors if all are non-positive."` | `"FitModel: Some sigma (error) values are non-positive. Absolute values will be used; if all are non-positive, errors will be ignored."` |

### Step 8 — Update tests

- Add tests in `tests/test_utils.py` (or a new `tests/test_logging.py`) that:
  1. Verify the `scatterbrain` logger has a NullHandler by default
  2. Verify `configure_logging()` adds a handler and sets the level
  3. Verify that analysis failures emit `WARNING`-level log records (use `pytest` + `caplog` fixture)
  4. Verify `AnalysisError` / `FittingError` / `ProcessingError` are raised at the expected call sites

Example test pattern using `caplog`:
```python
import logging
import pytest
from scatterbrain.analysis.guinier import guinier_fit

def test_guinier_warns_insufficient_points(caplog, empty_curve):
    with caplog.at_level(logging.WARNING, logger="scatterbrain"):
        result = guinier_fit(empty_curve)
    assert result is None
    assert "Insufficient data points" in caplog.text
```

---

## File Change Summary

| File | Changes |
|---|---|
| `scatterbrain/utils.py` | Add NullHandler registration; add `import logging` |
| `scatterbrain/__init__.py` | Add `configure_logging()` helper |
| `scatterbrain/core.py` | Add module logger; replace 1 `print()` with `logger.warning()`; raise `ProcessingError` at line 394 |
| `scatterbrain/io.py` | Add module logger; replace 6 `warnings.warn()` with `logger.warning()`; remove `import warnings` |
| `scatterbrain/analysis/guinier.py` | Add module logger; replace 1 `print()` with `logger.debug()`; replace 9 `warnings.warn()` with `logger.warning/debug()`; raise `AnalysisError`; fix typo; remove `import warnings` |
| `scatterbrain/analysis/porod.py` | Add module logger; replace 6 `warnings.warn()` with `logger.warning()`; raise `AnalysisError`; remove `import warnings` |
| `scatterbrain/modeling/fitting.py` | Add module logger; replace 9 `warnings.warn()` with `logger.warning()`; raise `FittingError`; remove `import warnings` |
| `scatterbrain/visualization.py` | Add module logger; replace 2 `warnings.warn()` with `logger.warning()`; remove `import warnings` |
| `tests/test_logging.py` (new) | Logger hierarchy test; `configure_logging()` test; `caplog`-based warning tests; custom exception raise tests |

---

## What Does NOT Change

- All `raise TypeError / ValueError` in `core.py` for input validation — these are correct Python conventions for bad arguments
- All `raise FileNotFoundError / IndexError` in `io.py` and `core.py` — these are correct Python conventions
- The `return None` soft-failure pattern in analysis/fitting functions — kept, but documented
- The `NotImplementedError` raises in `visualization.py` — correct placeholders
- No changes to public API signatures
