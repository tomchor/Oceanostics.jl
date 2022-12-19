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
                        stop_time=0.2,
                        )

χ = Oceanostics.FlowDiagnostics.IsotropicTracerVarianceDissipationRate(model, :c)
c2 = c^2

χ_int = Integral(χ, dims=(1,2)) # Can't integrate in all directions due to #2857
c2_int = Integral(c2, dims=(1,2))

outputs = (; χ, χ_int,
           c2, c2_int,
           )
simulation.output_writers[:tracer] = NetCDFOutputWriter(model, outputs;
                                                        filename = "tracer_diff.nc",
                                                        schedule = TimeInterval(1e-3),
                                                        overwrite_existing = true)

using Oceanostics: TimedProgressMessenger, SingleLineProgressMessenger
simulation.callbacks[:progress] = Callback(TimedProgressMessenger(), IterationInterval(10))

run!(simulation)
