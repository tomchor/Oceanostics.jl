using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging
using SeawaterPolynomials: RoquetEquationOfState, TEOS10EquationOfState

#+++ Common grid setup
# GPU detection
arch = has_cuda_gpu() ? GPU() : CPU()

# Grid size
N = 6

# Regular grid
regular_grid = RectilinearGrid(arch, size=(N, N, N), extent=(1, 1, 1))

# Stretched grid functions and setup
S = 0.99 # Stretching factor. Positive number ∈ (0, 1]
f_asin(k) = -asin(S*(2k - N - 2) / N)/π + 1/2
F1 = f_asin(1); F2 = f_asin(N+1)
z_faces(k) = ((F1 + F2)/2 - f_asin(k)) / (F1 - F2)

stretched_grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=z_faces)

# Common grids dictionary
grids = Dict("regular grid" => regular_grid,
             "stretched grid" => stretched_grid)
#---

#+++ Common functions
# Grid noise function
grid_noise(x, y, z) = randn()
#---

#+++ Common model configurations
# Common model kwargs
model_kwargs = (buoyancy = BuoyancyForce(BuoyancyTracer()),
                coriolis = FPlane(1e-4),
                tracers = :b)

# Common closures
closures = (ScalarDiffusivity(ν=1e-6, κ=1e-7),
            SmagorinskyLilly(),
            Smagorinsky(coefficient=DynamicCoefficient(averaging=(1, 2))),
            Smagorinsky(coefficient=DynamicCoefficient(averaging=LagrangianAveraging())),
            (ScalarDiffusivity(ν=1e-6, κ=1e-7), AnisotropicMinimumDissipation()),)

# Common buoyancy formulations
buoyancy_formulations = (nothing,
                         BuoyancyTracer(),
                         SeawaterBuoyancy())

# Common coriolis formulations
coriolis_formulations = (nothing,
                         FPlane(1e-4),
                         ConstantCartesianCoriolis(fx=1e-4, fy=1e-4, fz=1e-4))

# Extended buoyancy formulations (for some tests)
extended_buoyancy_formulations = (nothing,
                                  BuoyancyTracer(),
                                  SeawaterBuoyancy(),
                                  SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState()),
                                  SeawaterBuoyancy(equation_of_state=RoquetEquationOfState(:Linear)))

# Common model types
model_types = (NonhydrostaticModel,
               HydrostaticFreeSurfaceModel)
#---