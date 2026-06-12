using Test

group     = get(ENV, "TEST_GROUP", :all) |> Symbol

@testset "Oceanostics" begin
    if group == :vel_diagnostics || group == :all
        include("test_velocity_diagnostics.jl")
    end

    if group == :tracer_diagnostics || group == :all
        include("test_tracer_diagnostics.jl")
    end

    if group == :u_momentum_diagnostics || group == :all
        include("test_u_momentum_diagnostics.jl")
    end

    if group == :v_momentum_diagnostics || group == :all
        include("test_v_momentum_diagnostics.jl")
    end

    if group == :w_momentum_diagnostics || group == :all
        include("test_w_momentum_diagnostics.jl")
    end

    if group == :ke_diagnostics || group == :all
        include("test_kinetic_energy_equation.jl")
    end

    if group == :coarse_grained_ke_diagnostics || group == :all
        include("test_coarse_grained_kinetic_energy_equation.jl")
    end

    if group == :tke_diagnostics || group == :all
        include("test_turbulent_kinetic_energy_equation.jl")
    end

    if group == :pe_diagnostics || group == :all
        include("test_pe_diagnostics.jl")
    end

    if group == :active_tracer_diagnostics || group == :all
        include("test_active_tracer_diagnostics.jl")
    end

    if group == :tracer_variance_diagnostics || group == :all
        include("test_tracer_variance_diagnostics.jl")
    end

    if group == :general_flow_diagnostics || group == :all
        include("test_general_flow_diagnostics.jl")
    end

    if group == :canonical_flows || group == :all
        include("test_canonical_flows.jl")
    end

    if group == :progress_messengers || group == :all
        include("test_progress_messengers.jl")
    end

    if group == :filters || group == :all
        include("test_filters.jl")
    end

    if group == :perf_invariants || group == :all
        include("test_perf_invariants.jl")
    end
end
