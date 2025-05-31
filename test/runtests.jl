using Test

group     = get(ENV, "TEST_GROUP", :all) |> Symbol

@testset "Oceanostics" begin
    if group == :vel_diagnostics || group == :all
        include("test_velocity_diagnostics.jl")
    end

    if group == :tracer_diagnostics || group == :all
        include("test_tracer_diagnostics.jl")
    end

    if group == :ke_diagnostics || group == :all
        include("test_ke_diagnostics.jl")
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

    if group == :budgets || group == :all
        include("test_budgets.jl")
    end
end
