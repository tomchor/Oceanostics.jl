using Test
using CUDA: has_cuda_gpu

using Oceananigans
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging
using Oceanostics
using Oceanostics.ProgressMessengers

#+++ Default grids and functions
arch = has_cuda_gpu() ? GPU() : CPU()

N = 6
regular_grid = RectilinearGrid(arch, size=(N, N, N), extent=(1, 1, 1))
#---

#+++ Testing options
closures = (ScalarDiffusivity(ν=1e-6, κ=1e-7),
            SmagorinskyLilly(),
            Smagorinsky(coefficient=DynamicCoefficient(averaging=(1, 2))),
            Smagorinsky(coefficient=DynamicCoefficient(averaging=LagrangianAveraging())),
            (ScalarDiffusivity(ν=1e-6, κ=1e-7), AnisotropicMinimumDissipation()),)
#---

#+++ Testing functions
function test_progress_messenger(model, messenger)
    simulation = Simulation(model; Δt=1e-2, stop_iteration=10)
    simulation.callbacks[:progress] = Callback(messenger, IterationInterval(1))
    run!(simulation)
    return true
end
#---

@testset "ProgressMessenger tests" begin
    @info "    Testing ProgressMessengers"
    for closure in closures
        model = NonhydrostaticModel(grid = regular_grid;
                                    buoyancy = BuoyancyForce(BuoyancyTracer()),
                                    coriolis = FPlane(1e-4),
                                    tracers = :b,
                                    closure = closure)

        @info "        Testing BasicMessenger with $closure"
        model.clock.iteration = 0
        test_progress_messenger(model, BasicMessenger())

        @info "        Testing SingleLineMessenger with $closure"
        model.clock.iteration = 0
        @test test_progress_messenger(model, SingleLineMessenger())

        # Test that SingleLineMessenger is indeed a single line
        simulation = Simulation(model; Δt=1e-2, stop_iteration=1)
        msg = SingleLineMessenger(print=false)(simulation)
        @test countlines(IOBuffer(msg)) == 1

        @info "        Testing TimedMessenger with $closure"
        model.clock.iteration = 0
        @test test_progress_messenger(model, TimedMessenger())

        @info "        Testing custom progress messenger with $closure"
        model.clock.iteration = 0
        step_duration = StepDuration()
        progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false)
                                      + SimulationTime() + TimeStep() + MaxVelocities()
                                      + AdvectiveCFLNumber() + step_duration)(simulation)
        @test test_progress_messenger(model, progress)
    end
end
