using Oceananigans
using Oceananigans.Fields: @compute
using Oceanostics
using Random
using Statistics

import Oceananigans.Fields: Field
Field(a::Number) = a

function periodic_locations(N_locations, flip_z=true)
    Random.seed!(772)
    reflections = [-2;;-1;;0;;1;;2]

    x₀ = rand(N_locations) .+ reflections
    y₀ = rand(N_locations) .+ reflections
    z₀ = rand(N_locations) .+ reflections

    flip_z && (z₀ = -z₀)
    return x₀, y₀, z₀
end

function test_tracer_variance_budget(; N=16, rtol=0.01, closure = ScalarDiffusivity(ν=1, κ=1), regular_grid=true)

    if regular_grid
        grid = RectilinearGrid(topology=(Periodic, Flat, Periodic), size=(N,N), extent=(1,1))
    else
        S = 2
        zᵃᵃᶠ(k) = (tanh(S * (2 * (k - 1) / N - 1)) / tanh(S) - 1) / 2 # [-1.0, 0.0]
        grid = RectilinearGrid(topology=(Periodic, Flat, Bounded), size=(N,N), x=(0,1), z=zᵃᵃᶠ)
    end
    model = NonhydrostaticModel(grid=grid, tracers=:c, closure=closure)

    # A kind of convoluted way to create x-periodic, resolved initial noise
    σx = 4grid.Δxᶜᵃᵃ # x length scale of the noise
    σy = 4grid.Δyᵃᶜᵃ # x length scale of the noise
    σz = 4mean(zspacings(grid, Center(), Center(), Center())) # z length scale of the noise

    N_gaussians = 20 # How many Gaussians do we want sprinkled throughout the domain?
    Random.seed!(772)
    xₚ, yₚ, zₚ = periodic_locations(N_gaussians)

    resolved_noise(x, z) = sum(@. exp(-(x-xₚ)^2/σx^2 -(z-zₚ)^2/σz^2))
    set!(model, u=resolved_noise, c=resolved_noise)

    u, v, w = model.velocities
    c = model.tracers.c

    c.data.parent .-= mean(c)
    u.data.parent .-= mean(u)

    κ = diffusivity(model.closure, model.diffusivity_fields, Val(:c))
    @compute κ = κ isa Tuple ? Field(sum(κ)) : κ
    Δt = min(minimum_zspacing(grid)^2/maximum(κ)/10, minimum_zspacing(grid)/maximum(u) / 10)
    simulation = Simulation(model; Δt=Δt, stop_time=0.1)

    wizard = TimeStepWizard(cfl=0.1, diffusive_cfl=0.1)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

    ε  = KineticEnergyDissipationRate(model)
    @compute ∫εdV   = Field(Integral(ε))
    @compute ∫KEdV  = Field(Integral(KineticEnergy(model)))
    ∫∫εdVdt = Ref(0.0)
    ∫KEdV_t⁰ = parent(∫KEdV)[1,1,1]

    χ  = TracerVarianceDissipationRate(model, :c)
    @compute ∫χdV   = Field(Integral(χ))
    @compute ∫c²dV  = Field(Integral(c^2))
    ∫∫χdVdt = Ref(0.0)
    ∫c²dV_t⁰ = parent(∫c²dV)[1,1,1]

    function accumulate_χ(sim)
        compute!(∫εdV)
        ∫∫εdVdt[] += sim.Δt * only(interior(∫εdV, 1, 1, 1))

        compute!(∫χdV)
        ∫∫χdVdt[] += sim.Δt * only(interior(∫χdV, 1, 1, 1))
        return nothing
    end
    simulation.callbacks[:integrate_χ] = Callback(accumulate_χ)

    run!(simulation)

    compute!(∫KEdV)
    ∫KEdV_tᶠ = parent(∫KEdV)[1,1,1]
    ∫∫εdVdt_tᶠ = ∫KEdV_t⁰- ∫∫εdVdt[]
    abs_error = (abs(∫∫εdVdt_tᶠ - ∫KEdV_tᶠ)/∫KEdV_t⁰)
    @info "Error in KE decrease is $abs_error"
    @test abs_error < rtol

    compute!(∫c²dV)
    ∫c²dV_tᶠ = parent(∫c²dV)[1,1,1]
    ∫∫χdVdt_tᶠ = ∫c²dV_t⁰- ∫∫χdVdt[]
    abs_error = (abs(∫∫χdVdt_tᶠ - ∫c²dV_tᶠ)/∫c²dV_t⁰)
    @info "Error in c² decrease is $abs_error"
    @test abs_error < rtol

    return nothing
end
