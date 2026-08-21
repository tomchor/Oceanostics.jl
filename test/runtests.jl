using Test
import Dates
import Logging

# Prefix each log record with a timestamp ("[ HH:MM:SS Info: … ]" instead of "[ Info: … ]") so the
# per-section `@info` lines in the test output show when each stage ran.
function timestamped_metafmt(level, _module, _group, id, file, line)
    color, prefix, suffix = Logging.default_metafmt(level, _module, _group, id, file, line)
    return color, string(Dates.format(Dates.now(), "HH:MM:SS"), " ", prefix), suffix
end

# Only switch the logger in CI — local runs keep the plain format. CI (GitHub Actions) sets CI=true.
get(ENV, "CI", "false") == "true" &&
    Logging.global_logger(Logging.ConsoleLogger(stderr, Logging.Info; meta_formatter = timestamped_metafmt))

group     = get(ENV, "TEST_GROUP", :all) |> Symbol

#+++ Architecture reporting and enforcement
using CUDA: has_cuda_gpu

# Every test file picks `arch = has_cuda_gpu() ? GPU() : CPU()` for itself, which degrades to the
# CPU *silently* whenever CUDA.jl fails to initialise. That turns a broken GPU stack into a green
# build carrying no GPU coverage at all — precisely what the GPU pipeline exists to catch, and not
# hypothetical here: `nvidia-smi` can be perfectly happy while CUDA.jl dies creating a context on
# an sm_70 card. So announce the choice for every group, and refuse to run where a GPU is expected.
#
# `TEST_ARCHITECTURE` is the explicit control, matching the Oceananigans and OpenBoundaries
# pipelines on the same host: "GPU" demands one, "CPU" waives the demand. With it unset a GPU is
# required on hardware meant to have one — a Buildkite job, or the nautilus host. The agent
# container's hostname is its container ID rather than `nautilus`, so the Buildkite check is what
# covers CI and the hostname check covers running directly on the box.
#
# Note "CPU" only waives the *requirement*; the test files still choose by `has_cuda_gpu()`. To
# actually run on the CPU on a GPU machine, hide the devices with `CUDA_VISIBLE_DEVICES=""`.
requested_architecture = get(ENV, "TEST_ARCHITECTURE", "")
on_buildkite = haskey(ENV, "BUILDKITE")
on_nautilus  = startswith(gethostname(), "nautilus")

gpu_required = requested_architecture == "GPU" ||
               (requested_architecture != "CPU" && (on_buildkite || on_nautilus))
has_gpu = has_cuda_gpu()

if gpu_required && !has_gpu
    why = requested_architecture == "GPU" ? "TEST_ARCHITECTURE=GPU is set" :
          on_buildkite ? "this is a Buildkite job" : "this is the nautilus host"
    error("""
          A GPU is required here ($why), but `CUDA.has_cuda_gpu()` is false. Every test would fall
          back to the CPU and pass without exercising a single line of GPU code.

          Check `nvidia-smi` and `CUDA.versioninfo()` on this machine. To waive the requirement
          deliberately, set TEST_ARCHITECTURE=CPU.
          """)
end

architecture_label = has_gpu ? "GPU" : "CPU"
@info "Oceanostics test suite: running on $architecture_label (group: $group)"
#---

@testset "Oceanostics" begin
    if group == :vel_diagnostics || group == :all
        @info "Running test_velocity_diagnostics.jl on $architecture_label"
        include("test_velocity_diagnostics.jl")
    end

    if group == :tracer_diagnostics || group == :all
        @info "Running test_tracer_diagnostics.jl on $architecture_label"
        include("test_tracer_diagnostics.jl")
    end

    if group == :momentum_diagnostics || group == :all
        @info "Running test_momentum_diagnostics.jl on $architecture_label"
        include("test_momentum_diagnostics.jl")
    end

    if group == :ke_diagnostics || group == :all
        @info "Running test_kinetic_energy_equation.jl on $architecture_label"
        include("test_kinetic_energy_equation.jl")
    end

    if group == :filtered_ke_diagnostics || group == :all
        @info "Running test_filtered_kinetic_energy_equation.jl on $architecture_label"
        include("test_filtered_kinetic_energy_equation.jl")
    end

    if group == :subfilter_ke_diagnostics || group == :all
        @info "Running test_subfilter_kinetic_energy_equation.jl on $architecture_label"
        include("test_subfilter_kinetic_energy_equation.jl")
    end

    if group == :tke_diagnostics || group == :all
        @info "Running test_turbulent_kinetic_energy_equation.jl on $architecture_label"
        include("test_turbulent_kinetic_energy_equation.jl")
    end

    if group == :pe_diagnostics || group == :all
        @info "Running test_pe_diagnostics.jl on $architecture_label"
        include("test_pe_diagnostics.jl")
    end

    if group == :ape_diagnostics || group == :all
        @info "Running test_ape_diagnostics.jl on $architecture_label"
        include("test_ape_diagnostics.jl")
    end

    if group == :filtered_ape_diagnostics || group == :all
        @info "Running test_filtered_ape_diagnostics.jl on $architecture_label"
        include("test_filtered_ape_diagnostics.jl")
    end

    if group == :subfilter_ape_diagnostics || group == :all
        @info "Running test_subfilter_ape_diagnostics.jl on $architecture_label"
        include("test_subfilter_ape_diagnostics.jl")
    end

    if group == :active_tracer_diagnostics || group == :all
        @info "Running test_active_tracer_diagnostics.jl on $architecture_label"
        include("test_active_tracer_diagnostics.jl")
    end

    if group == :tracer_variance_diagnostics || group == :all
        @info "Running test_tracer_variance_diagnostics.jl on $architecture_label"
        include("test_tracer_variance_diagnostics.jl")
    end

    if group == :general_flow_diagnostics || group == :all
        @info "Running test_general_flow_diagnostics.jl on $architecture_label"
        include("test_general_flow_diagnostics.jl")
    end

    if group == :canonical_flows || group == :all
        @info "Running test_canonical_flows.jl on $architecture_label"
        include("test_canonical_flows.jl")
    end

    if group == :progress_messengers || group == :all
        @info "Running test_progress_messengers.jl on $architecture_label"
        include("test_progress_messengers.jl")
    end

    if group == :spatial_filters || group == :all
        @info "Running test_spatial_filters.jl on $architecture_label"
        include("test_spatial_filters.jl")
    end

    if group == :perf_invariants || group == :all
        @info "Running test_perf_invariants.jl on $architecture_label"
        include("test_perf_invariants.jl")
    end
end
