# CLAUDE.md

Oceanostics.jl provides diagnostics for [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) simulations.
It computes the terms of the tracer, tracer variance, momentum, kinetic energy, turbulent kinetic energy and potential energy budgets, their filtered and subfilter decompositions, and flow diagnostics such as Richardson and Rossby numbers and Ertel potential vorticity.
Area-specific guidance lives in `.claude/rules/*.md` and loads when a matching file is read; this file holds only what every session needs.

## Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run one test group (much faster for development)
TEST_GROUP=vel_diagnostics julia --project -e 'using Pkg; Pkg.test()'

# Instantiate the package
julia --project -e 'using Pkg; Pkg.instantiate()'

# Build the documentation, which runs every example
julia --project=docs docs/make.jl
```

The `TEST_GROUP` values are the `group == :…` symbols in `test/runtests.jl`.
`docs/build_single_page.jl` rebuilds one docs page from a REPL, in about a second after the first include; its header explains `PAGE` and `DOCTEST`.

## Map

`src/Oceanostics.jl` includes the modules in dependency order (the constraints are commented above the `include` list) and holds the top-level exports and the `@diagnostic_show` line that gives each diagnostic its display.

| Area | Key paths | Docs page | Rule file |
|---|---|---|---|
| Tracer, tracer variance and momentum budgets | `src/TracerEquation.jl`, `src/TracerVarianceEquation.jl`, `src/{U,V,W}MomentumEquation.jl` | `tracer_equation.md`, `tracer_variance_equation.md`, `momentum_equation.md` | none |
| Kinetic and turbulent kinetic energy budgets | `src/KineticEnergyEquation.jl`, `src/TurbulentKineticEnergyEquation.jl` | `kinetic_energy_equation.md`, `turbulent_kinetic_energy_equation.md` | none |
| Potential, background and available potential energy | `src/PotentialEnergyEquation.jl`, `src/BackgroundPotentialEnergyEquation.jl`, `src/*AvailablePotentialEnergyEquation.jl` | `*potential_energy_equation.md`, `validation/reference_state.md` | `potential-energy.md` |
| Spatial filters, filtered and subfilter budgets | `src/SpatialFilters/`, `src/Filtered*.jl`, `src/SubFilter*.jl` | `filters.md`, `filtered_*.md`, `subfilter_*.md` | `filtered-budgets.md` |
| Flow diagnostics | `src/FlowDiagnostics.jl` | `flow_diagnostics.md` | none |
| Progress messengers | `src/ProgressMessengers/` | `progress_messengers.md` | `progress-messengers.md` |
| Examples and the docs build | `docs/examples/`, `docs/make.jl`, `docs/build_single_page.jl` | `docs/README.md` | `docs-and-examples.md` |
| Tests and CI | `test/`, `.github/workflows/`, `.buildkite/gpu-pipeline.yml` | none | `testing-and-ci.md` |

## Before you touch…

- a test group: read `test/runtests.jl`, `.github/workflows/ci.yml` and the matrix in `.buildkite/gpu-pipeline.yml`; the `quality_assurance` group checks that the three agree.
- an example: read the header comments of `docs/make.jl`, which builds each example in its own CI job and reuses a generated page that is newer than its source.
- the reference state: read `docs/src/background_potential_energy_equation.md` §The reference state.
- the GPU pipeline: read the header of `.buildkite/gpu-pipeline.yml` and the architecture block at the top of `test/runtests.jl`.

## Rules

- Every diagnostic is an Oceananigans `KernelFunctionOperation` (KFO): an `@inline` kernel that computes the value at `(i, j, k)` with Oceananigans operators, a `const` alias `CustomKFO{<:typeof(kernel)}`, and constructors that take a `model` or the individual fields.
- A kernel's name ends in the location of its result (`_ccc`, `_ccf`, `_fff`); a constructor that supports a single location checks it with `validate_location`, and a dissipation rate checks its closure with `validate_dissipative_closure`.
- Functions and methods use `snake_case` and types, structs and modules `CamelCase`; a `CustomKFO` alias is a type, so its constructor methods stay `CamelCase`.
- Hand a kernel its advection scheme through `momentum_advection` or `tracer_advection`, never `model.advection`: the Oceananigans compat range covers releases where that field is a single scheme and releases where it is a `NamedTuple` (#309).
- Oceananigans model constructors take the grid positionally: `NonhydrostaticModel(grid; closure, tracers)`, not `NonhydrostaticModel(; grid, ...)`.
- Unicode identifiers follow the mathematical notation (ψ, ε, ν, ∂, ℑ).
- A code expression stays on one line up to 130 columns; prose in docstrings, comments and docs pages wraps at about 100 columns.
- Julia examples in docstrings and docs are `jldoctest` blocks, so they run as doctests, in script style (the code, a `# output` line, then the expected output) unless explicitly stated otherwise.
- Fold markers are `#+++ <title>` to open and `#---` to close, with exactly three `+` or `-` and no space after `#`; the `quality_assurance` group checks that they are well formed and balanced.

## Gotchas

- The docs build is also a test: five of the six examples carry hidden (`#hide`) `@test` checks, mostly budget residuals, so a change to a diagnostic can pass `Pkg.test()` and fail the Documentation workflow.
- Every GitHub Actions run is CPU-only, since `test/test_utils.jl` picks `GPU()` only when CUDA finds a device; GPU coverage comes from the Buildkite pipeline alone.
- A `TEST_GROUP` name is not its file name: `vel_diagnostics` runs `test_velocity_diagnostics.jl` and `ke_diagnostics` runs `test_kinetic_energy_equation.jl`.

## Keeping this file small

- A change updates the docs page and the comment beside the code; it does not add prose here.
- A rule file gains a line only when a session got that thing wrong without it: one sentence saying what to do, the why, and the issue or PR number.
- This file changes only for a command, a convention, or a repo-wide rule; prefer a pointer to a page over a list that will drift.
- One sentence per line in this file and in every rule file, with no wrapping at 100 columns, no narrative and no history; a rule file stays around 30 lines of rules, frontmatter aside.
- Nothing visible in `Project.toml` or the file tree is restated here or in a rule file.
