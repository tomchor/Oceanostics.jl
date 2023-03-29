using Oceananigans.TurbulenceClosures: HorizontalFormulation, VerticalFormulation
using Oceananigans.Fields: @compute
using Random
using Statistics

function periodic_locations(N_locations, flip_z=true)
    Random.seed!(772)
    reflections = [-2;;-1;;0;;1;;2]

    x₀ = rand(N_locations) .+ reflections
    y₀ = rand(N_locations) .+ reflections
    z₀ = rand(N_locations) .+ reflections

    flip_z && (z₀ = -z₀)
    return x₀, y₀, z₀
end

function test_tracer_variance_budget(; N=4, κ=1, rtol=0.01, closure = ScalarDiffusivity(HorizontalFormulation(), κ=κ))

    grid = RectilinearGrid(topology=(Periodic, Periodic, Periodic), size=(N,N,N), extent=(1,1,1))
    model = NonhydrostaticModel(grid=grid, tracers=:c, closure=closure,
                                auxiliary_fields=(; ∫∫χdVdt=0.0, ∫χdV_prev=0.0))

    # A kind of convoluted way to create x-periodic, resolved initial noise
    σx = 4grid.Δxᶜᵃᵃ # x length scale of the noise
    σy = 4grid.Δyᵃᶜᵃ # x length scale of the noise
    σz = 4grid.Δzᵃᵃᶜ # z length scale of the noise

    N_gaussians = 20 # How many Gaussians do we want sprinkled throughout the domain?
    Random.seed!(772)
    xₚ, yₚ, zₚ = periodic_locations(N_gaussians)

    resolved_noise(x, y, z) = sum(@. exp(-(x-xₚ)^2/σx^2 -(y-yₚ)^2/σy^2 -(z-zₚ)^2/σz^2))
    set!(model, c=resolved_noise)
    c = model.tracers.c
    c.data.parent .-= mean(c)

    simulation = Simulation(model; Δt=grid.Δxᶜᵃᵃ^2/κ/20, stop_time=0.1)

    χ  = Oceanostics.FlowDiagnostics.TracerVarianceDissipationRate(model, :c)

    @compute ∫χdV   = Field(Integral(χ))
    @compute ∫c²dV  = Field(Integral(c^2))

    ∫c²dV_t⁰ = parent(∫c²dV)[1,1,1]
    function accumulate_χ(sim)
        increment = sim.Δt * model.auxiliary_fields.∫χdV_prev
        model.auxiliary_fields = (; model.auxiliary_fields..., ∫∫χdVdt = model.auxiliary_fields.∫∫χdVdt + increment)

        compute!(∫χdV)
        model.auxiliary_fields = (; model.auxiliary_fields..., ∫χdV_prev = parent(∫χdV)[1,1,1])
        return nothing
    end
    simulation.callbacks[:integrate_χ] = Callback(accumulate_χ)

    run!(simulation)

    compute!(∫c²dV)
    ∫c²dV_tᶠ = parent(∫c²dV)[1,1,1]
    ∫∫χdVdt_tᶠ = ∫c²dV_t⁰- model.auxiliary_fields.∫∫χdVdt
    abs_error = (abs(∫∫χdVdt_tᶠ - ∫c²dV_tᶠ)/∫c²dV_t⁰)

    @info "Error in c² decrease is $abs_error"
    @test abs_error < rtol

    return nothing
end
