---
paths:
  - "test/**"
  - ".github/workflows/**"
  - ".buildkite/**"
---
# Tests and CI

## Groups and pipelines
`TEST_GROUP` values are the `group == :…` symbols in `test/runtests.jl`; every group has a job in `.github/workflows/ci.yml` and all but two have an entry in the `.buildkite/gpu-pipeline.yml` matrix, which the `quality_assurance` group checks.
`perf_invariants` and `quality_assurance` stay off the GPU pipeline: the first builds `CPU()` grids and asserts wall-clock ratios, the second only reads files; the comments under the pipeline's matrix give the details.
Buildkite allows 25 elements per matrix dimension and 50 jobs per matrix; the GPU matrix is one-dimensional, so a 26th group needs a second matrix step, and `quality_assurance` fails before Buildkite does.
On Buildkite or the nautilus host `runtests.jl` refuses to run without a GPU; `TEST_ARCHITECTURE=CPU` waives the demand, and `CUDA_VISIBLE_DEVICES=""` is what actually runs the suite on the CPU there.
The GPU pipeline needs no `CUDA_Runtime_jll` pin and sets `JULIA_CUDA_USE_COMPAT=false` only as a conservative default; its header records the measurements behind both.

## Writing tests
Build a test from the grids, closures, buoyancy and Coriolis formulations and model types in `test/test_utils.jl` rather than defining new ones.
Put new grids on `arch = has_cuda_gpu() ? GPU() : CPU()` so the same file runs on the CPU under Actions and on the GPU under Buildkite; only `test_perf_invariants.jl` builds `CPU()` grids on purpose.
`test_momentum_diagnostics.jl` runs one set of test functions over the `DIRECTIONS` table (u, v, w); add a momentum term there once, not per component.
A new `src/` module adds a representative KFO to `test_perf_invariants.jl`, which checks that per-cell evaluation is type-stable and allocates nothing.
