# Repository guidance

These rules apply to SNOPT.jl and OptimizationSNOPT.jl work.

## Priorities

1. Preserve numerical correctness and public API compatibility.
2. Make installation and execution instructions copyable.
3. Prefer clear, direct code over clever abstractions.
4. Explain required domain terms. Do not invent terminology.

## Technical writing

- Use Simplified Technical English where practical.
- Keep prose sentences under 20 words when practical.
- Give each sentence one main idea.
- Define an abbreviation or specialized term on first use.
- Use the same term for the same concept across both packages.
- Lead instructions with the expected result.
- State the working directory and prerequisites for commands.
- Use tables for repeated mappings, statistics, or comparable results.
- Keep definitions, option names, and table columns consistent.
- Avoid promotional language, filler, and unexplained jargon.
- Prefer short examples that users can copy and run.

Code, equations, links, and API identifiers do not follow the sentence limit.
Longer sentences are acceptable when splitting them would reduce accuracy.

## Design by contract

- State relevant preconditions for public functions.
- State return values and observable side effects.
- State process-wide and workspace invariants near the code that enforces them.
- Validate unsafe inputs at public boundaries.
- Use specific exception types and actionable messages.
- Document array shapes, mutation rules, units, and callback return meanings.
- Add loop-invariant comments only when the invariant is not obvious.
- Do not add contract machinery when a plain Julia check is clearer.

Important package invariants include:

- SNOPT owns one active Fortran workspace per Julia process.
- Workspace creation and solves must remain serialized.
- Callback arrays must have the documented sizes and storage order.
- Bounds and derivative arrays must match the problem dimensions.
- Option normalization occurs before values reach the SNOPT library.

## Testing

- Use test-driven development for behavior changes and bug fixes.
- First write a test that fails for the expected reason.
- Add the smallest implementation that makes the test pass.
- Use table-driven or generated Julia cases for property-style coverage.
- Cover valid inputs, boundaries, invalid inputs, and important invariants.
- Describe behavior scenarios with Given, When, and Then when useful.
- Give test sets names that describe observable behavior.
- Test real package behavior. Avoid tests that only inspect source text.
- Run library-free tests and licensed solver tests when available.
- Use Aqua and Julia's normal analysis tools before adding dependencies.
- Use formal verification only when a maintained Julia tool supports the code.
- Do not add Python-only tools such as Hypothesis, Nagini, or Axiomander.

## Documentation checks

Before completion:

1. Run both package test suites.
2. Build both Documenter sites.
3. Run the documented examples when the SNOPT library is available.
4. Check commands, links, option names, and cross-package terminology.
5. Check for placeholders, undefined terms, and avoidable long sentences.
6. Review the diff for unrelated API or behavior changes.

If a licensed SNOPT library is unavailable, report each skipped check.
