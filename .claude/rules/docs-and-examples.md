---
paths:
  - "docs/**"
  - ".github/workflows/documentation.yml"
---
# Docs and examples

## Examples
An example is a Literate script that `docs/make.jl` runs with `execute = true`, one per job of the Documentation workflow; a new one goes in both the `examples` list of `make.jl` and the matrix of `.github/workflows/documentation.yml`.
`#hide` lines run but do not render, and the `@test` lines among them are regression tests; keep them when rewriting an example's prose.
Budget tendencies come from `TimeDerivative`; list each one directly in the writer's `outputs`, since `add_dependencies!` dispatches on the output itself and a `TimeDerivative` inside a composite operation never gets its update callback (#313).
The first record of a budget writer is zero, since the derivative has no earlier state to difference against, so post-processing starts at record 2 (`nb = 2:length(ds["time"])`).
The derivative is a backward difference centered at `t - Δt/2` while the source terms sit at `t`, so a budget closes only to `O(Δt)`; look there first when a residual tolerance starts to look marginal.
`make.jl` reuses a generated page that is newer than its `.jl`, so delete `docs/src/generated` or touch the example to force a rebuild.

## Building locally
`docs/build_single_page.jl` rebuilds one page from a REPL; an example page is staged from `docs/src/generated`, so regenerate it with Literate first, as the script's header explains.
A KFO's `show` prints argument types relative to `Main`, so a local `doctest(Oceanostics)` needs `using Oceananigans` first, as `make.jl` has; without it about 100 doctests fail on fully qualified names.
Method counts drift between Oceananigans versions, which is why `make.jl` filters `with \d+ methods?`; keep that filter in any new `makedocs` or `doctest` call.
`docs/` is one self-contained environment reached through `[sources]`, so it resolves Oceananigans itself and a stale `docs/Manifest.toml` can pin an old one: doctests then fail on `show` output like the model's advection scheme, and the fix is to delete the manifest rather than edit the expected output (#317).
Run the doctests with `julia --project=docs -e 'using Documenter, Oceananigans, Oceanostics; doctest(Oceanostics; doctestfilters = [r"with \d+ methods?"])'`, which needs no scratch environment now that `Oceanostics` and `Oceananigans` are both docs dependencies.
