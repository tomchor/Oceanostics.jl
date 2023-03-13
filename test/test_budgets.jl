using Oceananigans.TurbulenceClosures: HorizontalFormulation, VerticalFormulation
using Oceananigans.Fields: @compute

function test_tracer_variance_budget(; N=4, κ=2, rtol=0.01)
    closure = ScalarDiffusivity(HorizontalFormulation(), κ=κ)

    grid = RectilinearGrid(topology=(Periodic, Periodic, Periodic),
                           size=(N,N,N), extent=(1,1,1))
    model = NonhydrostaticModel(grid=grid, tracers=:c, closure=closure, auxiliary_fields=(; ∫∫χdVdt=0.0))

    # A kind of convoluted way to create x-periodic, resolved initial noise
    σx = 4grid.Δxᶜᵃᵃ # x length scale of the noise
    σy = 4grid.Δyᵃᶜᵃ # x length scale of the noise
    σz = 4grid.Δzᵃᵃᶜ # z length scale of the noise

    N_gaussians = 20 # How many Gaussians do we want sprinkled throughout the domain?
    x₀ = grid.Lx * rand(N_gaussians); y₀ = grid.Ly * rand(N_gaussians); z₀ = -grid.Lz * rand(N_gaussians) # Locations of the Gaussians

    xₚ = x₀ .+ (grid.Lx .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection
    yₚ = y₀ .+ (grid.Ly .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection
    zₚ = z₀ .+ (grid.Lz .* [-2;;-1;;0;;1;;2]) # Make that noise periodic by "infinite" horizontal reflection

    resolved_noise(x, y, z) = sum(@. exp(-(x-xₚ)^2/σx^2 -(y-yₚ)^2/σy^2 -(z-zₚ)^2/σz^2))
    set!(model, c=resolved_noise)

    simulation = Simulation(model; Δt=grid.Δxᶜᵃᵃ^2/κ/20, stop_time=0.1)

    c = model.tracers.c
    χ  = Oceanostics.FlowDiagnostics.IsotropicTracerVarianceDissipationRate(model, :c)

    @compute ∫χdV   = Field(Integral(χ))
    @compute ∫c²dV  = Field(Integral(c^2))

    ∫c²dV_t⁰ = parent(∫c²dV)[1,1,1]
    function accumulate_χ(sim)
        compute!(∫χdV)
        increment = sim.Δt * parent(∫χdV)[1,1,1]
        model.auxiliary_fields = (; ∫∫χdVdt = model.auxiliary_fields.∫∫χdVdt + increment)
        return nothing
    end
    simulation.callbacks[:integrate_χ] = Callback(accumulate_χ)

    run!(simulation)

    compute!(∫c²dV)
    ∫c²dV_tᶠ = parent(∫c²dV)[1,1,1]
    ∫∫χdVdt_tᶠ = ∫c²dV_t⁰- model.auxiliary_fields.∫∫χdVdt
    abs_error = (abs(∫∫χdVdt_tᶠ - ∫c²dV_tᶠ)/∫c²dV_tᶠ)

    @info "Error in c² decrease is $abs_error"
    @test ≈(∫∫χdVdt_tᶠ, ∫c²dV_tᶠ, rtol=rtol)

    return nothing
end
