using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans: fill_halo_regions!
using Oceanostics: free_surface_displacement
using Oceananigans.TurbulenceClosures.Smagorinskys: LagrangianAveraging
using Oceananigans.TurbulenceClosures: DynamicCoefficient, LagrangianAveraging
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

# Volume and area reductions as scalars, i.e. without scalar indexing on GPUs
volume_integral(op) = sum(∫dV(op))
volume_mean(op)     = sum(Field(Average(op)))  # volume-weighted over the whole domain
area_integral(op)   = sum(Field(Integral(op, dims=(1, 2))))  # of a surface diagnostic, at (Center, Center, Nothing)

# Interior values copied to the CPU, so that comparisons index a plain `Array` on either architecture
cpu_interior(f::Field) = Array(interior(f))
cpu_interior(op) = Array(interior(Field(op)))

# Subtract the depth mean from a `HydrostaticFreeSurfaceModel`'s `u` and `v`. The depth-integrated flow
# is what moves the free surface, so without it the `w` diagnosed from continuity vanishes at the surface
# as well as at the bottom, and the domain is closed the way a nonhydrostatic one with rigid walls is.
function remove_barotropic_flow!(model)
    u, v, _ = model.velocities
    u′ = Field(u - Field(Average(u, dims=3)))
    v′ = Field(v - Field(Average(v, dims=3)))
    set!(model, u = u′, v = v′)
    return nothing
end
#---

#+++ Common model configurations
# Common model kwargs
model_kwargs = (buoyancy = BuoyancyForce(BuoyancyTracer()),
                coriolis = FPlane(f=1e-4),
                tracers = :b)

# Common closures
closures = (ScalarDiffusivity(ν=1e-6, κ=1e-7),
            SmagorinskyLilly(),
            DynamicSmagorinsky(averaging=(1, 2)),
            DynamicSmagorinsky(averaging=LagrangianAveraging()),
            (ScalarDiffusivity(ν=1e-6, κ=1e-7), AnisotropicMinimumDissipation()),)

# Common buoyancy formulations
buoyancy_formulations = (nothing,
                         BuoyancyTracer(),
                         SeawaterBuoyancy())

# Common coriolis formulations
coriolis_formulations = (nothing,
                         FPlane(f=1e-4),
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

# A model of `model_type` with a free surface, an `ImplicitFreeSurface` unless another is given: the
# hydrostatic model's default on these grids, and the one the nonhydrostatic model supports, whose
# default is a rigid lid
free_surface_model(model_type, grid; free_surface=ImplicitFreeSurface(), kwargs...) =
    model_type(grid; free_surface, kwargs...)

# Set the free surface's displacement `η`, a `ZFaceField` windowed to the top face, halos included.
# `set!` on the model does not reach it on a `NonhydrostaticModel`, and on that model `set!` projects
# the velocities it is given and in doing so steps `η` by one second of the surface `w`, so a test sets
# the velocities first and `η` afterwards.
function set_free_surface!(model, η_function)
    η = free_surface_displacement(model.free_surface)
    set!(η, η_function)
    fill_halo_regions!(η)
    return nothing
end
#---