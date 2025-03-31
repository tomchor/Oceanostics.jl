using Test

group     = get(ENV, "TEST_GROUP", :all) |> Symbol

function test_mixed_layer_depth(grid; 
                                buoyancy = SeawaterBuoyancy(equation_of_state=BoussinesqEquationOfState(LinearRoquetSeawaterPolynomial(), 1000),
                                                            constant_salinity=0),
                                zₘₓₗ = 0.5,
                                δρ = 0.125,
                                ∂z_T = - δρ / zₘₓₗ / buoyancy.equation_of_state.seawater_polynomial.R₀₁₀)
    boundary_conditions = FieldBoundaryConditions(grid, (Center, Center, Center); top = GradientBoundaryCondition(∂z_T))

    T = CenterField(grid; boundary_conditions)

    mld = MixedLayerDepth(grid, buoyancy, (; T))

    @test isinf(mld[1, 1])
    
    set!(T, (x, y, z) -> z * ∂z_T+10)
    fill_halo_regions!(T)

    @test mld[1, 1] ≈ -zₘₓₗ

    return nothing
end

@testset "Oceanostics" begin
    if group == :vel_diagnostics || group == :all
        include("test_velocity_diagnostics.jl")
    end

    if group == :ke_diagnostics || group == :all
        include("test_ke_diagnostics.jl")
    end

    if group == :pe_diagnostics || group == :all
        include("test_pe_diagnostics.jl")
    end

    if group == :tracer_diagnostics || group == :all
        include("test_tracer_diagnostics.jl")
    end

    if group == :canonical_flows || group == :all
        include("test_canonical_flows.jl")
    end

    test_mixed_layer_depth(grid)
    if group == :progress_messengers || group == :all
        include("test_progress_messengers.jl")
    end

    if group == :budgets || group == :all
        include("test_budgets.jl")
    end
end
