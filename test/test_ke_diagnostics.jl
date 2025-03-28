using Test
using CUDA: has_cuda_gpu, @allowscalar

using Oceananigans
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: @compute
using Oceananigans.TurbulenceClosures: ThreeDimensionalFormulation
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging

using Oceanostics
using Oceanostics: BuoyancyProductionTerm, AdvectionTerm

#+++ Default grids
arch = has_cuda_gpu() ? GPU() : CPU()

N = 6
regular_grid = RectilinearGrid(arch, size=(N, N, N), extent=(1, 1, 1))

S = .99 # Stretching factor. Positive number ∈ (0, 1]
f_asin(k) = -asin(S*(2k - N - 2) / N)/π + 1/2
F1 = f_asin(1); F2 = f_asin(N+1)
z_faces(k) = ((F1 + F2)/2 - f_asin(k)) / (F1 - F2)

stretched_grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=z_faces)

grid_noise(x, y, z) = randn()
#---

#+++ Test options
model_kwargs = (buoyancy = BuoyancyForce(BuoyancyTracer()),
                coriolis = FPlane(1e-4),
                tracers = :b)

closures = (ScalarDiffusivity(ν=1e-6, κ=1e-7),
            SmagorinskyLilly(),
            Smagorinsky(coefficient=DynamicCoefficient(averaging=(1, 2))),
            Smagorinsky(coefficient=DynamicCoefficient(averaging=LagrangianAveraging())),
            (ScalarDiffusivity(ν=1e-6, κ=1e-7), AnisotropicMinimumDissipation()),)

coriolis_formulations = (nothing,
                         FPlane(1e-4),
                         ConstantCartesianCoriolis(fx=1e-4, fy=1e-4, fz=1e-4))

grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)

model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---

#+++ Test functions
function test_ke_calculation(model)
    u, v, w = model.velocities
    U = Field(Average(u, dims=(2, 3)))
    V = Field(Average(v, dims=(2, 3)))
    W = Field(Average(w, dims=(2, 3)))

    ke_op = KineticEnergy(model)
    @test ke_op isa AbstractOperation
    ke_c = Field(ke_op)
    @test all(interior(compute!(ke_c)) .≈ 0)

    op = TurbulentKineticEnergy(model, U=U, V=V, W=W)
    @test op isa AbstractOperation
    tke_c = Field(op)
    @test all(interior(compute!(tke_c)) .≈ 0)

    return nothing
end
function test_shear_production_terms(model)
    u, v, w = model.velocities
    U = Field(Average(u, dims=(2, 3)))
    V = Field(Average(v, dims=(2, 3)))
    W = Field(Average(w, dims=(2, 3)))

    op = XShearProductionRate(model, u, v, w, U, V, W)
    @test op isa AbstractOperation
    XSP = Field(op)
    @test all(interior(compute!(XSP)) .≈ 0)

    op = XShearProductionRate(model; U=U, V=V, W=W)
    @test op isa AbstractOperation
    XSP = Field(op)
    @test all(interior(compute!(XSP)) .≈ 0)

    U = Field(Average(u, dims=(1, 3)))
    V = Field(Average(v, dims=(1, 3)))
    W = Field(Average(w, dims=(1, 3)))

    op = YShearProductionRate(model; U=U, V=V, W=W)
    @test op isa AbstractOperation
    YSP = Field(op)
    @test all(interior(compute!(YSP)) .≈ 0)

    op = YShearProductionRate(model, u, v, w, U, V, W)
    @test op isa AbstractOperation
    YSP = Field(op)
    @test all(interior(compute!(YSP)) .≈ 0)


    U = Field(Average(u, dims=(1, 2)))
    V = Field(Average(v, dims=(1, 2)))
    W = Field(Average(w, dims=(1, 2)))

    op = ZShearProductionRate(model, u, v, w, U, V, W)
    @test op isa AbstractOperation
    ZSP = Field(op)
    @test all(interior(compute!(ZSP)) .≈ 0)

    op = ZShearProductionRate(model; U=U, V=V, W=W)
    @test op isa AbstractOperation
    ZSP = Field(op)
    @test all(interior(compute!(ZSP)) .≈ 0)

end

function test_pressure_term(model)
    u⃗∇p = PressureRedistributionTerm(model)
    @test u⃗∇p isa AbstractOperation
    @test compute!(Field(u⃗∇p)) isa Field

    u⃗∇pNHS = PressureRedistributionTerm(model, pressure=model.pressures.pNHS)
    @test u⃗∇pNHS isa AbstractOperation
    @test compute!(Field(u⃗∇pNHS)) isa Field

    # Test calculation with a hydrostatic pressure separation
    model2 = NonhydrostaticModel(grid=model.grid, hydrostatic_pressure_anomaly=CenterField(model.grid))
    u⃗∇p_from_model2 = PressureRedistributionTerm(model2)
    @test u⃗∇p_from_model2 isa AbstractOperation
    @test compute!(Field(u⃗∇p_from_model2)) isa Field

    return nothing
end

function test_momentum_advection_term(grid; model_type=NonhydrostaticModel)
    model = model_type(; grid)
    C₁ = 2; C₂ = 3
    set!(model, u=(x, y, z) -> C₁*y, v=C₂)

    ADV = AdvectionTerm(model)
    @compute ADV_field = Field(ADV)
    @test ADV isa AbstractOperation
    @test ADV_field isa Field

    # Test excluding the grid boundaries
    @test Array(interior(ADV_field, 1, 2:grid.Ny-1, 1)) ≈ collect(C₁^2 * C₂ * grid.yᵃᶜᵃ[2:grid.Ny-1])

    return nothing
end

function test_ke_dissipation_rate_terms(grid; model_type=NonhydrostaticModel, closure=ScalarDiffusivity(ν=1))
    model = model_type(; grid, closure, buoyancy=BuoyancyTracer(), tracers=:b)

    if !(model.closure isa Tuple) || all(isa.(model.closure, ScalarDiffusivity{ThreeDimensionalFormulation}))
        ε_iso = IsotropicKineticEnergyDissipationRate(model; U=0, V=0, W=0)
        ε_iso_field = compute!(Field(ε_iso))
        @test ε_iso isa AbstractOperation
        @test ε_iso_field isa Field
    end

    dudz = 2
    set!(model, u=(x, y, z) -> dudz*z)

    ε = KineticEnergyDissipationRate(model)
    ε_field = compute!(Field(ε))
    @test ε isa AbstractOperation
    @test ε_field isa Field

    εp = KineticEnergyDissipationRate(model; U=Field(Average(model.velocities.u, dims=(1,2))))
    εp_field = compute!(Field(εp))
    @test εp isa AbstractOperation
    @test εp_field isa Field

    idxs = (model.grid.Nx÷2, model.grid.Ny÷2, model.grid.Nz÷2)

    if closure isa Tuple
        @compute ν_field = Field(sum(viscosity(closure, model.diffusivity_fields)))
    else
        ν_field = viscosity(closure, model.diffusivity_fields)
    end

    rtol = zspacings(grid, Center()) isa Number ? 1e-12 : 0.06 # less accurate for stretched grid

    @allowscalar begin
        true_ε = (ν_field isa Field ? getindex(ν_field, idxs...) : ν_field) * dudz^2
        @test isapprox(getindex(ε_field,  idxs...), true_ε, rtol=rtol, atol=eps())
        @test isapprox(getindex(εp_field, idxs...), 0.0,    rtol=rtol, atol=eps())
    end

    ε = KineticEnergyStressTerm(model)
    ε_field = compute!(Field(ε))
    @test ε isa AbstractOperation
    @test ε_field isa Field

    set!(model, u=grid_noise, v=grid_noise, w=grid_noise, b=grid_noise)
    @compute ε̄ₖ = Field(Average(KineticEnergyDissipationRate(model)))
    @compute ε̄ₖ₂= Field(Average(KineticEnergyStressTerm(model)))


    if model isa NonhydrostaticModel
        @test Array(interior(ε̄ₖ, 1, 1, 1)) ≈ Array(interior(ε̄ₖ₂, 1, 1, 1))

        ε = KineticEnergyTendency(model)
        @compute ε_field = Field(ε)
        @test ε isa AbstractOperation
        @test ε_field isa Field

        @compute ∂ₜKE = Field(Average(TracerVarianceTendency(model, :b)))
    end

    return nothing
end

function test_ke_forcing_term(grid; model_type=NonhydrostaticModel)
    Fᵘ_func(x, y, z, t, u) = -u
    Fᵛ_func(x, y, z, t, v) = -v
    Fʷ_func(x, y, z, t, w) = -w

    Fᵘ = Forcing(Fᵘ_func, field_dependencies = :u)
    Fᵛ = Forcing(Fᵛ_func, field_dependencies = :v)
    Fʷ = Forcing(Fʷ_func, field_dependencies = :w)

    model = model_type(; grid, forcing = (u=Fᵘ, v=Fᵛ, w=Fʷ))
    set!(model, u=grid_noise, v=grid_noise, w=grid_noise)

    ε = KineticEnergyForcingTerm(model)
    @compute ε_field = Field(ε)
    @test ε isa AbstractOperation
    @test ε_field isa Field

    @compute ε_truth = Field(@at (Center, Center, Center) (-model.velocities.u^2 -model.velocities.v^2 -model.velocities.w^2))

    @test isapprox(Array(interior(ε_field, 1, 1, 1)), Array(interior(ε_truth, 1, 1, 1)), rtol=1e-12, atol=eps())

    return nothing
end

function test_buoyancy_production_term(grid; model_type=NonhydrostaticModel)
    model = model_type(grid=grid, buoyancy=BuoyancyTracer(), tracers=:b)
    w₀ = 2; b₀ = 3
    set!(model, w=w₀, b=b₀, enforce_incompressibility=false)

    wb = BuoyancyProductionTerm(model)
    @compute wb_field = Field(wb)
    @test wb isa AbstractOperation
    @test wb_field isa Field
    @test Array(interior(wb_field, 1, 1, 2)) .== w₀ * b₀

    w′ = Field(model.velocities.w - Field(Average(model.velocities.w)))
    b′ = Field(model.tracers.b - Field(Average(model.tracers.b)))
    w′b′ = BuoyancyProductionTerm(model, velocities=(u=model.velocities.u, v=model.velocities.v, w=w′), tracers=(b=b′,))
    @compute w′b′_field = Field(w′b′)
    @test w′b′ isa AbstractOperation
    @test w′b′_field isa Field
    @test .≈(Array(interior(w′b′_field, 1, 1, 2)), 0, rtol=1e-12, atol=1e-13) # less accurate for stretched grid

    return nothing
end
#---

@testset "KE/TKE diagnostics tests" begin
    @info "  Testing KE/TKE diagnostics"
    for (grid_class, grid) in zip(keys(grids), values(grids))
        @info "    with $grid_class"
        for model_type in model_types
            @info "      with $model_type"
            for closure in closures
                @info "        with closure $(summary(closure))"
                model = model_type(; grid, closure, model_kwargs...)

                if model isa NonhydrostaticModel
                    @info "          Testing pressure terms"
                    test_pressure_term(model)

                    @info "          Testing buoyancy production term"
                    test_buoyancy_production_term(grid; model_type)
                end

                @info "          Testing energy dissipation rate terms"
                test_ke_dissipation_rate_terms(grid; model_type, closure)

                if model_type == NonhydrostaticModel
                    @info "          Testing advection terms"
                    test_momentum_advection_term(grid; model_type)

                    @info "          Testing forcing terms"
                    test_ke_forcing_term(grid; model_type)

                @info "          Testing velocity-only diagnostics"
                test_shear_production_terms(model)
                end
            end
        end

        @info "        Testing input validation for dissipation rates"
        invalid_closures = [HorizontalScalarDiffusivity(ν=1e-6, κ=1e-7),
                            VerticalScalarDiffusivity(ν=1e-6, κ=1e-7),
                            (ScalarDiffusivity(ν=1e-6, κ=1e-7), HorizontalScalarDiffusivity(ν=1e-6, κ=1e-7))]

        for closure in invalid_closures
            model = NonhydrostaticModel(grid = regular_grid; model_kwargs..., closure)
            @test_throws ErrorException IsotropicKineticEnergyDissipationRate(model; U=0, V=0, W=0)
        end

    end
end
