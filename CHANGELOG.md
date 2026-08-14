# Changelog

## 0.3.0

### Breaking

- `SnoptResult` gained a `basis` field, and `SnoptB`/`SnoptC` gained a trailing
  `nS` field. The previous positional constructors for the problem types still
  work and default `nS` to `0`, so existing code that builds them by hand is
  unaffected.
- `snopt` rejects `start = "Warm"` or `"Hot"` without a `basis`, and rejects a
  `basis` on a cold start. Both were previously accepted and silently ignored.

### Added

- `SnoptBasis` and the `basis` keyword of `snopt`, which make `start = "Warm"`
  and `start = "Hot"` reuse the previous basis instead of silently starting cold.
- `snlog` now works on `SnoptA` solves, through SNOPT's `snKerA` kernel.
- Objective, gradient, and constraint callbacks accept any callable object, not
  only `Function` subtypes.

### Fixed

- `snoptb!` and `snoptc!` reject Jacobians whose shape does not match the
  problem, instead of letting SNOPT read past the end of `locJ`.
- Concurrent solves are serialized internally rather than corrupting one
  another's workspace. Workspace sizing, creation, option application, and the
  solve are held as a single transaction.
- Library discovery requires the `f_*` interface symbols, so a `libsnopt7` built
  without the C interface is reported at load time instead of at the first solve.
- The scratch summary file used for suppressed output is created eagerly in a
  writable location, so an unwritable temporary directory no longer leaves the
  session broken with every later solve returning status 82.

## 0.2.2

### Fixed

- `snopta!` start modes: `snOptA` uses SNOPT's own convention (0 cold, 1 basis
  file, 2 warm, 3 hot), which differs from the `snOptB`/`snOptC` wrappers.
  `"Warm"` previously requested a basis-file start.
- `NaN` is rejected in `x0`, bounds, and constraint bounds; `x0` must be finite.
- The `Float64` form of `set_option!` rejects non-finite values.
- The preflight evaluation clamps `x0` into the variable bounds, matching
  SNOPT's own projection before its first evaluation.
- `free!` no longer calls `snEnd` for a workspace that never reached
  `f_sninitx`, and `initialize` frees the workspace if the defaults reset fails.
- Library discovery warns and falls back when `SNOPTDIR` is set but unloadable,
  and finally searches the system loader's default paths.
- Inform code 33 is reported as `:Superbasics_Limit_Too_Small` rather than as an
  iteration limit.

### Added

- `THIRD_PARTY_NOTICE.md` recording SNOPT's commercial license, as the Julia
  General registry requires for wrappers of restrictive libraries.
- Aqua is part of the test suite, and runs before the solver tests so that
  library-free CI still checks packaging hygiene.
