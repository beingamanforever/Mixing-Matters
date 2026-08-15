# Mixing Matters agent guidelines

## Instruction priority

Read this file, `CLAUDE.md`, the relevant runbook, and `paper/EXPERIMENT-SOURCE-OF-TRUTH.md` before editing or running an experiment.
Preserve repository-specific instructions and existing public contracts unless the task explicitly requires a change.
Never manually edit `CHANGELOG.md` or files marked as generated.
Never add an agent as a commit co-author.
Do not use the em dash character.
Write each complete sentence on its own line when substantially editing long Markdown files.

## Core development rules

- Target Python 3.10 or newer.
- Add type hints to all code.
- Give public APIs clear docstrings.
- Use docstrings to document modules, classes, functions, non-obvious constraints, and invariants.
- Keep functions focused and small.
- Follow existing patterns and module structure exactly when adding workers, pipelines, or commands.
- Prefer configuration-driven behavior over hardcoded parameters.
- Use PEP 8 naming: `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Use f-strings for string formatting.
- Prefix handler function names with `handle`.
- Prefer constants over functions when no behavior is required.
- Use early returns to avoid unnecessary nesting.
- Keep logic DRY without introducing unnecessary abstractions.
- Prefer functional, immutable, and stateless code when it improves clarity without adding verbosity.
- Define composing functions before their component functions.
- Mark known issues in existing code with a `TODO:` comment.
- Keep changes minimal and related to the task.
- Balance file organization with simplicity and the scale of the project.

## Development philosophy

- Simplicity: write straightforward code and avoid clever solutions.
- Readability: make behavior and intent easy to understand.
- Performance: consider runtime and memory costs without sacrificing readability.
- Maintainability: optimize for robust, scalable, long-term evolution rather than short-term development cost.
- Testability: design behavior that can be validated with realistic inputs.
- Reusability: reuse existing components and experiment code before adding new machinery.
- Less code means less debt: minimize the code footprint.
- Clean logic: keep core logic pure where practical and push I/O and framework details to the edges.

## Required workflow

Before editing, inspect the relevant code, tests, callers, conventions, branch history, runbooks, and artifacts.
For a bug, first reproduce the user-visible failure as closely as possible and identify one evidence-backed mechanism before changing code.
Build iteratively, starting with the smallest complete behavior and verifying it before adding complexity.
Run focused tests frequently with realistic inputs, and create a controlled test environment when a component cannot be validated directly.
Never hide failures, weaken tests, or add silent fallbacks to make checks pass.
Treat lint failures, test failures, and flaky tests as real issues.
After implementation, review the final diff and run relevant formatting, linting, type checks, tests, builds, and end-to-end checks.
Remove only temporary files created by the current task and never delete unknown user files.

## Experiment protocol

Execute roadmap work one gated phase at a time.
Reuse validated code and environments from earlier phases when their revisions and assumptions match.
Pin the repository revision, model revision, dataset checksum, configuration, seed, runtime, and hardware in every run manifest.
Write raw records immutably and never overwrite a completed result file.
Keep prompts, distractors, question bundles, decoding settings, and scoring fixed unless the phase explicitly varies one of them.
Use the question as the resampling unit and preserve every ten-position bundle.
Separate measured results, interpretation, planned work, and missing validation.
Do not claim a control or ablation passed until its outputs, labels, process state, logs, GPU state, record counts, and summary artifacts have been inspected.
Commit and push each completed phase before starting the next phase.
Record the exact branch, commit, commands, environment, artifacts, checks, and remaining limitations in a concise phase report.

## Agent orchestration

Use a planner for non-trivial, multi-file, unclear, architectural, or high-risk work.
Use an explorer when paths, callers, ownership, tests, commands, or control flow are unclear.
Use a researcher when current primary evidence can materially affect a technical decision.
Give implementers one focused outcome with owned files, forbidden scope, invariants, acceptance criteria, and verification commands.
Use an independent reviewer after meaningful code changes and a verifier after implementation and every material repair.
Parallelize only independent work with disjoint ownership and no shared contract.
Do not declare completion until every acceptance criterion has evidence and review and verification are complete.
