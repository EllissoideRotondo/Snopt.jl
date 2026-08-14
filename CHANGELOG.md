# Changelog

## 0.3.0

### Breaking

- Minimum Julia version is now 1.10 (1.9 reached end of life).
- The high-level `snopt` rejects `start = "Hot"` with an `ArgumentError`.
  SNOPT's hot start reuses factorization state stored inside the previous
  solve's workspace; the high-level entry point builds a fresh workspace per
  call, so a hot start read uninitialized memory and crashed. Use
  `start = "Warm"` with a `basis`, or drive the low-level interface with one
  workspace.
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
- `snstop`, SNOPT's `snSTOP` termination hook, is available on `snopt`, `snopt!`,
  `snopta!`, `snoptb!`, and `snoptc!`. It delivers a `SnoptStopEvent` per major
  iteration - everything `snlog` reports plus the objective gradient, Jacobian
  values, row multipliers, reduced costs, and reduced gradient - and returning
  `false` from it stops the solve with `:User_Requested_Stop`. Use it for custom
  termination criteria.
- Objective, gradient, and constraint callbacks accept any callable object, not
  only `Function` subtypes.

### Fixed

- `snoptb!` and `snoptc!` reject Jacobians whose shape does not match the
  problem, instead of letting SNOPT read past the end of `locJ`.
- Concurrent solves are serialized internally rather than corrupting one
  another's workspace. Workspace sizing, creation, option application, and the
  solve are held as a single transaction.
- Library discovery probes every `f_*` interface symbol the package calls
  (solvers, kernels, options, specs, memory), so a `libsnopt7` built without
  the C interface — or with only part of it — is reported at load time instead
  of at the first solve. The resolved library path is normalized to an
  absolute path.
- `set_option!` accepts any `Integer`/`Real` value width as documented, not
  just `Int`/`Float64`.
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
