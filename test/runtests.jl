using Test

group     = get(ENV, "TEST_GROUP", :all) |> Symbol

@testset "Oceanostics" begin
    if group == :diagnostics || group == :all
        include("test_diagnostics.jl")
    end

    if group == :ke_diagnostics || group == :all
        include("test_ke_diagnostics.jl")
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
