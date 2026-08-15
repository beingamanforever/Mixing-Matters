# Mixing Matters development guidelines

Read and follow `AGENTS.md` before any repository change or experiment.
The rules below are mandatory summaries for Claude-compatible agents and do not replace the detailed experiment and orchestration requirements in `AGENTS.md`.

## Code quality

- Target Python 3.10 or newer.
- Add type hints to all code.
- Give public APIs clear docstrings and document modules, classes, functions, constraints, and invariants with docstrings where appropriate.
- Keep functions focused and small.
- Follow existing patterns and module structure exactly, including for new workers and pipelines.
- Prefer configuration-driven behavior over hardcoded parameters.
- Use PEP 8 naming, `PascalCase` class names, `UPPER_SNAKE_CASE` constants, and f-strings.
- Prefix handler function names with `handle`.
- Prefer early returns, descriptive names, constants where behavior is unnecessary, and DRY logic.
- Prefer functional, immutable, and stateless code when it stays concise.
- Define composing functions before their components.
- Use `TODO:` for known issues in existing code.
- Keep changes minimal and related to the task.

## Development philosophy

Prioritize simplicity, readability, robustness, scalability, maintainability, testability, and reuse.
Consider performance without sacrificing readability.
Minimize the code footprint because less code creates less debt.
Keep core logic clean and push framework, file, network, and process details to the edges.
Balance file organization with simplicity and project scale.

## Workflow and experiments

Inspect relevant instructions, plans, code, tests, callers, history, runbooks, and artifacts before editing.
For bug fixes, reproduce the end-user failure before changing code.
Build iteratively and run realistic focused tests frequently.
Create controlled test environments for components that are difficult to validate directly.
Never hide failures, weaken tests, or add silent fallbacks.
Review the final diff and run relevant formatting, linting, type checks, tests, builds, and end-to-end checks.
Execute experiments one gated phase at a time, save immutable raw outputs and manifests, then commit and push the completed phase before starting the next.
Reuse prior experiment code only after checking its pinned revision and assumptions.
Separate measured evidence from interpretation, proposed work, and missing validation.
Preserve user files and remove only task-owned temporary files.

Do not use the em dash character.
Never add an agent as a commit co-author.
Never manually edit `CHANGELOG.md` or generated files.
Write each complete sentence on its own line when substantially editing long Markdown files.
