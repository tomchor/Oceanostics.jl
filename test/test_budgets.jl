using Oceananigans.TurbulenceClosures: HorizontalFormulation, VerticalFormulation
using NCDatasets

function test_tracer_variance_budget(; N=4, κ=2, rtol=0.01)
    closure = ScalarDiffusivity(HorizontalFormulation(), κ=κ)

    grid = RectilinearGrid(topology=(Periodic, Periodic, Periodic),
                           size=(N,N,N), extent=(1,1,1))
    model = NonhydrostaticModel(grid=grid, tracers=:c, closure=closure)

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

    simulation = Simulation(model; Δt=grid.Δxᶜᵃᵃ^2/κ/20, stop_time=0.05)

    c = model.tracers.c
    χ  = Oceanostics.FlowDiagnostics.IsotropicTracerVarianceDissipationRate(model, :c)
    c² = c^2

    ∫χdV   = Integral(χ)
    ∫c²dV  = Integral(c²)

    simulation.output_writers[:tracer] = NetCDFOutputWriter(model, (; ∫χdV, ∫c²dV),
                                                            filename = "test_tracer_variance_budget.nc",
                                                            schedule = IterationInterval(1),
                                                            overwrite_existing = true)
    run!(simulation)

    ds = NCDataset(simulation.output_writers[:tracer].filepath, "r")
    ∫∫χdVdt_final = ds["∫c²dV"][1] .- (cumsum(ds["∫χdV"]) * simulation.Δt)[end]
    ∫c²dV_final   = ds["∫c²dV"][end]
    close(ds)
    @test ≈(∫∫χdVdt_final, ∫c²dV_final, rtol=rtol)

    return nothing
end
