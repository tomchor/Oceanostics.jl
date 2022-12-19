using Oceananigans
using Oceanostics
using Statistics

N = 4
κ = 1

grid = RectilinearGrid(topology=(Periodic, Periodic, Periodic),
                       size=(N,N,N), extent=(1,1,1))
model = NonhydrostaticModel(grid=grid, tracers=:c,
                            closure=ScalarDiffusivity(κ=κ))

c = model.tracers.c
cᵢ = randn(size(c)...)
cᵢ .-= mean(cᵢ)
set!(model, c=cᵢ)

simulation = Simulation(model; Δt=grid.Δxᶜᵃᵃ^2/κ/100,
                        stop_time=0.1,
                        )
simulation.callbacks[:progress] = Callback(TimedProgressMessenger(), IterationInterval(10))
wizard = TimeStepWizard(max_change=1.02, diffusive_cfl=0.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

χ = Oceanostics.FlowDiagnostics.IsotropicTracerVarianceDissipationRate(model, :c)
ε_q = @at (Center, Center, Center) κ * (∂x(∂x(c^2)) + ∂y(∂y(c^2)) + ∂z(∂z(c^2)))
c2 = c^2

χ_int = Integral(χ, dims=(1,2)) # Can't integrate in all directions due to #2857
ε_q_int = Integral(ε_q, dims=(1,2)) # Can't integrate in all directions due to #2857
c2_int = Integral(c2, dims=(1,2))

outputs = (; χ, χ_int,
           ε_q, ε_q_int,
           c2, c2_int,
           )
simulation.output_writers[:tracer] = NetCDFOutputWriter(model, outputs;
                                                        filename = "tracer_diff.nc",
                                                        schedule = TimeInterval(1e-5),
                                                        overwrite_existing = true)

using Oceanostics: TimedProgressMessenger, SingleLineProgressMessenger

run!(simulation)
