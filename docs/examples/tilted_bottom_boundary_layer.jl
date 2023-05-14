# # Tilted bottom boundary layer example
#
# This example simulates a two-dimensional oceanic bottom boundary layer
# in a domain that's tilted with respect to gravity. We simulate the perturbation
# away from a constant along-slope (y-direction) velocity constant density stratification.
# This perturbation develops into a turbulent bottom boundary layer due to momentum
# loss at the bottom boundary modeled with a quadratic drag law.
# 
#
# First let's make sure we have all required packages installed.
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, Rasters, CairoMakie"
# ```
#
# ## Model and simulation setup
#
# We create a 2D ``x, z`` grid with 64² cells and finer resolution near the bottom:

using Oceananigans
using Oceananigans.Units

Lx = 200meters
Lz = 100meters
Nx = 64
Nz = 64

## Creates a grid with near-constant spacing `refinement * Lz / Nz`
## near the bottom:
refinement = 1.8 # controls spacing near surface (higher means finer spaced)
stretching = 10  # controls rate of stretching at bottom 

h(k) = (Nz + 1 - k) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
z_faces(k) = - Lz * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(topology = (Periodic, Flat, Bounded), size = (Nx, Nz),
                       x = (0, Lx), z = z_faces)

# ## Tilting the domain
#
# We use a domain that's tilted with respect to gravity by

θ = 3 # degrees

# so that ``x`` is the along-slope direction, ``z`` is the across-sloce direction that
# is perpendicular to the bottom, and the unit vector anti-aligned with gravity is

ĝ = [sind(θ), 0, cosd(θ)]

# Changing the vertical direction impacts both the `gravity_unit_vector`
# for `Buoyancy` as well as the `rotation_axis` for Coriolis forces,

buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = -ĝ)
coriolis = ConstantCartesianCoriolis(f = 1e-4, rotation_axis = ĝ)

# where we have used a constant Coriolis parameter ``f = 10⁻⁴ \rm{s}⁻¹``.
# The tilting also affects the kind of density stratified flows we can model.
# In particular, a constant density stratification in the tilted
# coordinate system

@inline constant_stratification(x, y, z, t, p) = p.N² * (x * p.ĝ[1] + z * p.ĝ[3])

# is _not_ periodic in ``x``. Thus we cannot explicitly model a constant stratification
# on an ``x``-periodic grid such as the one used here. Instead, we simulate periodic
# _perturbations_ away from the constant density stratification by imposing
# a constant stratification as a `BackgroundField`,

B_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = 1e-5))

# where ``N² = 10⁻⁵ \rm{s}⁻¹`` is the background buoyancy gradient.

# ## Bottom drag
#
# We impose bottom drag that follows Monin-Obukhov theory.
# We include the background flow in the drag calculation,
# which is the only effect the background flow enters the problem,

V∞ = 0.1 # m s⁻¹
z₀ = 0.1 # m (roughness length)
κ = 0.4 # von Karman constant
z₁ = znodes(grid, Center())[1] # Closest grid center to the bottom
cᴰ = (κ / log(z₁ / z₀))^2 # Drag coefficient

@inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * u
@inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * (v + p.V∞)

drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)

# ## Create the `NonhydrostaticModel`
#
# We are now ready to create the model. We create a `NonhydrostaticModel` with an
# `UpwindBiasedFifthOrder` advection scheme, a `RungeKutta3` timestepper,
# and a constant viscosity and diffusivity. Here we use a smallish value of ``10^{-4} m² s⁻¹``.

closure = ScalarDiffusivity(ν=1e-4, κ=1e-4)

model = NonhydrostaticModel(; grid, buoyancy, coriolis, closure,
                            timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder(),
                            tracers = :b,
                            boundary_conditions = (u=u_bcs, v=v_bcs),
                            background_fields = (; b=B_field))

# ## Create and run a simulation
#
# We are now ready to create the simulation. We begin by setting the initial time step
# conservatively, based on the smallest grid size of our domain and set-up a 

using Oceananigans.Units

simulation = Simulation(model, Δt = 0.5 * minimum_zspacing(grid) / V∞, stop_time = 2days)

# We use `TimeStepWizard` to adapt our time-step and print a progress message,

using Printf

wizard = TimeStepWizard(max_change=1.1, cfl=0.7)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

# ## Model diagnostics
#
# We set-up a progress messenger using the `SimpleProgressMessenger`, which, as the name suggests,
# displays simple information about the simulation

using Oceanostics

progress = SimpleProgressMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))


# We can also define some useful diagnostics for of the flow, starting with the `RichardsonNumber`

Ri = RichardsonNumber(model, add_background=true)
PV = ErtelPotentialVorticity(model, add_background=true)
Ro = RossbyNumber(model, add_background=true)

# Now we write these quantities, along with `b`, to a NetCDF:

output_fields = (; Ri, Ro, PV, model.tracers.b)
filename = "tilted_bottom_boundary_layer"
simulation.output_writers[:nc] = NetCDFOutputWriter(model, output_fields,
                                                    filename = joinpath(@__DIR__, filename),
                                                    schedule = TimeInterval(1),
                                                    overwrite_existing = true)

# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# Now we'll read the results using Rasters.jl, which works somewhat similarly to Python's Xarray and
# can speed-up the work the workflow

using Rasters

ds = RasterStack(simulation.output_writers[:nc].filepath)

# We now use Makie to create the figure and its axes


