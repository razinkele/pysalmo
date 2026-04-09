# Development Environment

## Conda Environment
- **Use conda environment `shiny` for ALL Python work in this project**
- Run commands with: `conda run -n shiny python ...`
- Run tests with: `conda run -n shiny python -m pytest tests/ -v`
- Run benchmarks with: `conda run -n shiny python benchmarks/bench_full.py`
- **Available packages in `shiny` env:**
  - JAX 0.7.2 (GPU-capable)
  - meshio 5.3.5 (FEM mesh reading)
  - numba 0.64.0
  - numpy, scipy, pandas, geopandas, mesa, pydantic, hypothesis
- **Project path:** `C:\Users\arturas.baziukas\OneDrive - ku.lt\HORIZON_EUROPE\inSTREAM\instream-py`
- **GitHub (PySALMO):** https://github.com/razinkele/pysalmo
- **GitHub (inSTREAM-py, upstream):** https://github.com/razinkele/instream-py

---

# Shell Command Rules

When writing bash commands, strictly follow these rules to avoid triggering security permission prompts:

## No backslash-escaped whitespace
- Never escape spaces with backslashes (e.g. My\ File.txt)
- Always use double quotes around paths containing spaces: "My File.txt"
- Quote all variable expansions and file paths as a default habit

## No $() command substitution
- Never use $() or backtick substitution in inline bash commands
- Instead, run the inner command first as a separate bash call, read its output, then use the literal value in the next command
- When possible, rewrite using pipes and xargs (e.g. pgrep python | xargs kill instead of kill $(pgrep python))

## No multi-line commands with comments
- Never include # comments inside inline multi-line bash commands
- If a command requires multiple lines with comments, write it to a temporary script file in /tmp/ and execute that file instead
- Prefer collapsing short multi-line commands into a single line joined with &&

## No cd && git compound commands
- Never chain cd <path> && git <cmd>
- Use git -C <path> <cmd> instead

## General principles
- Prefer multiple simple, single-line bash calls over one complex compound command
- For anything that cannot be expressed as a clean single-line command, write a temporary .sh script and run it
- Always use double quotes around paths, variables, and arguments containing spaces or special characters

---

# Development Style

## Non-interactive, automated workflow
- **Do NOT ask clarifying questions** — make reasonable decisions and proceed
- **Do NOT pause for confirmation** between steps — execute the full task end-to-end
- **Do NOT propose plans and wait for approval** — just implement directly
- **Do NOT summarize what you're about to do** — do it
- If multiple valid approaches exist, pick the simplest one and go
- If something fails, diagnose and fix it yourself before asking for help
- Only interrupt the user when: credentials/secrets are needed, or the task is fundamentally unclear
- Run tests automatically after making changes — don't ask whether to run them
- Commit only when explicitly asked — but stage and verify without prompting

## Auto-approve all safe operations
- **git commands** (add, commit, status, diff, log, branch, checkout, push, pull, fetch, merge, rebase) — always run without asking
- **cd / directory navigation** — always proceed without asking
- **File reads, writes, edits, deletes** — always proceed without asking
- **Running tests, linters, formatters** — always proceed without asking
- **Installing packages** — always proceed without asking
- **Creating/deleting branches** — always proceed without asking
- **Any reversible local operation** — always proceed without asking
- Treat ALL tool calls as pre-approved unless they would delete user data that cannot be recovered

---

## PySALMO Architecture

### LifeStage enum
- `src/instream/agents/life_stage.py` — NEVER use magic numbers for life_history
- Values: FRY=0, PARR=1, SPAWNER=2, SMOLT=3, OCEAN_JUVENILE=4, OCEAN_ADULT=5, RETURNING_ADULT=6

### inSALMO parity (freshwater)
- Migration as 4th activity in habitat selection (`behavior.py` ~line 1108)
- 4x habitat selection passes per day (dawn/day/dusk/night) for anadromous species
- Per-substep resource regeneration between passes
- `survival^step_length` fitness exponentiation
- Forward condition projection: `mean_condition_survival()` in `survival.py`
- Adult holding: `skip_indices` in `select_habitat_and_activity` excludes spawners
- Features (opt-in via config): stochastic migration, two-piece condition-survival, spawn-cell noise, growth fitness

### Marine domain
- Optional — enabled by `marine:` config section
- `MarineDomain` in `src/instream/domains/marine.py` — orchestrates marine step
- Environmental drivers in `src/instream/io/env_drivers/` — StaticDriver for testing
- Modules: `marine_growth.py`, `marine_survival.py`, `marine_fishing.py`, `smoltification.py`, `marine_migration.py`

### Key tests
- Run all: `micromamba run -n shiny python -m pytest tests/ -v`
- Marine: `micromamba run -n shiny python -m pytest tests/test_marine_*.py tests/test_smolt*.py tests/test_full_lifecycle.py -v`
- Parity: `micromamba run -n shiny python -m pytest tests/test_performance_parity.py tests/test_insalmo_parity.py -v`
- Performance parity is slow (~8 min) — runs full 913-day Example A simulation

### Known parity gap
- Outmigrants: 52% of NetLogo (21k vs 41k)
- Root cause: drift food formula differs (Python: `conc*area*depth`; NetLogo: `86400*area*depth*velocity*conc/regen_dist`)
- Fix requires: drift formula change + parameter recalibration + NetLogo-style fitness function
- See: `docs/plans/2026-04-09-condition-projection-plan.md`
