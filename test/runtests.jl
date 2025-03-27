group     = get(ENV, "TEST_GROUP", :all) |> Symbol

@testset "Oceanostics" begin
    if group == :diagnostics || group == :all
        include("test_diagnostics.jl")
    end

    if group == :progress_messengers || group == :all
        include("test_progress_messengers.jl")
    end

    if group == :budgets || group == :all
        include("test_budgets.jl")
    end
end
