## # Tilted bottom boundary layer example
##
## This example is based on the similar [Oceananigans
## example](https://clima.github.io/OceananigansDocumentation/stable/generated/tilted_bottom_boundary_layer/)
## and simulates a two-dimensional oceanic bottom boundary layer in a domain that's tilted with
## respect to gravity. We simulate the perturbation away from a constant along-slope
## (y-direction) velocity constant density stratification.  This perturbation develops into a
## turbulent bottom boundary layer due to momentum loss at the bottom boundary.
## 
##
## First let's make sure we have all required packages installed.
##
## ```julia
## using Pkg
## pkg"add Oceananigans, Oceanostics, Rasters, CairoMakie"
## ```
##
## ## Grid
##
## We start by creating a ``x, z`` grid with 64² cells and finer resolution near the bottom:
#
##using Oceananigans
##using Oceananigans.Units
##
##Lx = 200meters
##Lz = 100meters
##Nx = 64
##Nz = 64
##
##refinement = 1.8 # controls spacing near surface (higher means finer spaced)
##stretching = 10  # controls rate of stretching at bottom 
##
##h(k) = (Nz + 1 - k) / Nz
##ζ(k) = 1 + (h(k) - 1) / refinement
##Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
##z_faces(k) = - Lz * (ζ(k) * Σ(k) - 1)
##
##grid = RectilinearGrid(topology = (Periodic, Flat, Bounded), size = (Nx, Nz),
##                       x = (0, Lx), z = (0, Lz))
#
### Note that, with the `z` faces defined as above, the spacings near the bottom are approximately
### constant, becoming progressively coarser moving up.
###
### ## Tilting the domain
###
### We use a domain that's tilted with respect to gravity by
##
##θ = 5; # degrees
##
### so that ``x`` is the along-slope direction, ``z`` is the across-sloce direction that
### is perpendicular to the bottom, and the unit vector anti-aligned with gravity is
##
##ĝ = [sind(θ), 0, cosd(θ)]
##
### Changing the vertical direction impacts both the `gravity_unit_vector` for `Buoyancy` as well as
### the `rotation_axis` for Coriolis forces,
##
##buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = -ĝ)
##
##f₀ = 1e-4/second
##coriolis = ConstantCartesianCoriolis(f = f₀, rotation_axis = ĝ)
##
### The tilting also affects the kind of density stratified flows we can model. The simulate an
### environment that's uniformly stratified, with a stratification frequency
##
##N² = 1e-5/second^2;
##
### In a tilted coordinate, this can be achieved with
##
##@inline constant_stratification(x, z, t, p) = p.N² * (x * p.ĝ[1] + z * p.ĝ[3]);
##
### However, this distribution is _not_ periodic in ``x`` and can't be explicitly modelled on an
### ``x``-periodic grid such as the one used here. Instead, we simulate periodic _perturbations_ away
### from the constant density stratification by imposing a constant stratification as a
### `BackgroundField`,
##
##B_field = BackgroundField(constant_stratification, parameters=(; ĝ, N²))
##
### ## Bottom drag
###
### We impose bottom drag that follows Monin-Obukhov theory and include the background flow in the
### drag calculation, which is the only effect the background flow has on the problem
##
##V∞ = 0.1meters/second
##z₀ = 0.1meters # (roughness length)
##κ = 0.4 # von Karman constant
##z₁ = znodes(grid, Center())[1] # Closest grid center to the bottom
##cᴰ = (κ / log(z₁ / z₀))^2 # Drag coefficient
##
##@inline drag_u(x, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * u
##@inline drag_v(x, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * (v + p.V∞)
##
##drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))
##drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))
##
##u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
##v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)
##
### ## Create model and simulation
###
### We are now ready to create the model. We create a `NonhydrostaticModel` with an
### `UpwindBiasedFifthOrder` advection scheme, a `RungeKutta3` timestepper, and a constant viscosity
### and diffusivity.
##
##closure = ScalarDiffusivity(ν=2e-4, κ=2e-4)
##
##model = NonhydrostaticModel(; grid, buoyancy, coriolis, closure,
##                            timestepper = :RungeKutta3,
##                            advection = UpwindBiasedFifthOrder(),
##                            tracers = :b,
##                            boundary_conditions = (u=u_bcs, v=v_bcs),
##                            background_fields = (; b=B_field))
##
##noise(x, z) = 1e-3 * randn() * exp(-(10z)^2/grid.Lz^2)
##set!(model, u=noise, w=noise)
##
### The bottom-intensified noise above should accelerate the emergence of turbulence close to the
### wall.
###
### We are now ready to create the simulation. We begin by setting the initial time step
### conservatively, based on the smallest grid size of our domain and set-up a 
##
##using Oceananigans.Units
##
##simulation = Simulation(model, Δt = 0.5 * minimum_zspacing(grid) / V∞, stop_time = 12hours)
##
### We use `TimeStepWizard` to maximize Δt
##
##wizard = TimeStepWizard(max_change=1.1, cfl=0.7)
##simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))
#
#
#
#
##
### ## Model and simulation setup
##
### We begin by creating a model with an isotropic diffusivity and fifth-order advection on a `xz`
### 128² grid using a buoyancy `b` as the active scalar. We'll work here with nondimensional
### quantities.
#
#using Oceananigans
#
#N = 128
#L = 10
#grid = RectilinearGrid(size=(N, N), x=(-L/2, +L/2), z=(-L/2, +L/2), topology=(Periodic, Flat, Bounded))
#
#model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3,
#                            advection = UpwindBiasedFifthOrder(),
#                            closure = ScalarDiffusivity(ν=2e-5, κ=2e-5),
#                            buoyancy = BuoyancyTracer(), tracers = :b)
#
## We use hyperbolic tangent functions for the initial conditions and set the maximum Richardson
## number below the threshold of 1/4. We also add some grid-scale small-amplitude noise to `u` to
## kick the instability off:
#
#
#noise(x, z) = 2e-2 * randn()
#shear_flow(x, z) = tanh(z) + noise(x, z)
#
#Ri₀ = 0.1; h = 1/4
#stratification(x, z) = h * Ri₀ * tanh(z / h)
#
#set!(model, u=shear_flow, b=stratification)
#
##
## Next create an adaptive-time-step simulation using the model above:
#
#simulation = Simulation(model, Δt=0.1, stop_time=100)
#
#wizard = TimeStepWizard(cfl=0.8, max_Δt=1)
#simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))
#
#
## ## Model diagnostics
##
## We set-up a progress messenger using the `TimedMessenger`, which displays, among other
## information, the time step duration
#
#using Oceanostics
#
#progress = ProgressMessengers.TimedMessenger()
#simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))
#
#
## We can also define some useful diagnostics for of the flow, starting with the `RichardsonNumber`
#
#Ri = RichardsonNumber(model)
#
## We also set-up the `QVelocityGradientTensorInvariant`, which is usually used for visualizing
## vortices in the flow: 
#Q = QVelocityGradientTensorInvariant(model)
#
## Q is one of the velocity gradient tensor invariants and it measures the amount of vorticity versus
## the strain in the flow and, when it's positive, indicates a vortex. This method of vortex
## visualization is called the [Q-criterion](https://tinyurl.com/mwv6fskc).
##
## Let's also keep track of the amount of buoyancy mixing by measuring the buoyancy
## variance dissipation rate and diffusive term. When volume-integrated, these two quantities should
## be equal.
#
#∫χᴰ = Integral(TracerVarianceDissipationRate(model, :b))
# Integral(TracerVarianceDiffusiveTerm(model, :b))
#
#
## Now we write these quantities, along with `b`, to a NetCDF:
#
#output_fields = (; Ri, model.tracers.b,)
#filename = "kelvin_helmholtz"
#simulation.output_writers[:nc] = NetCDFOutputWriter(model, output_fields,
#                                                    filename = joinpath(@__DIR__, filename),
#                                                    schedule = TimeInterval(1),
#                                                    overwrite_existing = true)
#
#
## ## Run the simulation and process results
##
## To run the simulation:
#
#run!(simulation)
#
## Now we'll read the results using Rasters.jl, which works somewhat similarly to Python's Xarray and
## can speed-up the work the workflow
#
#using Rasters
#
#ds = RasterStack(simulation.output_writers[:nc].filepath)
#
## We now use Makie to create the figure and its axes
#
#using CairoMakie
#
#set_theme!(Theme(fontsize = 24))
#fig = Figure()
#
#kwargs = (xlabel="x", ylabel="z", height=150, width=250)
#ax1 = Axis(fig[2, 1]; title = "Ri", kwargs...)
#ax2 = Axis(fig[2, 2]; title = "Q", kwargs...)
#ax3 = Axis(fig[2, 3]; title = "b", kwargs...);
#
## Next we use `Observable`s to lift the values and plot heatmaps and their colorbars
#
#n = Observable(1)
#
#Riₙ = @lift set(ds.Ri[Ti=$n, yC=Near(0)], :xC => X, :zF => Z)
#hm1 = heatmap!(ax1, Riₙ; colormap = :bwr, colorrange = (-1, +1))
#Colorbar(fig[3, 1], hm1, vertical=false, height=8)
#
#bₙ = @lift set(ds.b[Ti=$n, yC=Near(0)], :xC => X, :zC => Z)
#hm3 = heatmap!(ax3, bₙ; colormap = :balance, colorrange = (-2.5e-2, +2.5e-2))
#Colorbar(fig[3, 3], hm3, vertical=false, height=8);
#
## We now plot the time evolution of our integrated quantities
#
#axb = Axis(fig[4, 1:3]; xlabel="Time", height=100)
#times = dims(ds, :Ti)
##axislegend(position=:lb, labelsize=14)
#
## Now we mark the time by placing a vertical line in the bottom panel and adding a helpful title
#
#tₙ = @lift times[$n]
#vlines!(axb, tₙ, color=:black, linestyle=:dash)
#
#title = @lift "Time = " * string(round(times[$n], digits=2))
#fig[1, 1:3] = Label(fig, title, fontsize=24, tellwidth=false);
#
## Finally, we adjust the figure dimensions to fit all the panels and record a movie
#
#resize_to_layout!(fig)
#
#@info "Animating..."
#record(fig, filename * ".mp4", 1:length(times), framerate=10) do i
#       n[] = i
#end
#
## ![](kelvin_helmholtz.mp4)
##
## Similarly to the kinetic energy dissipation rate (see the [Two-dimensional turbulence example](@ref two_d_turbulence_example)), 
## `TracerVarianceDissipationRate` and `TracerVarianceDiffusiveTerm` are implemented
## with a energy-conserving formulation, which means that (for `NoFlux` boundary conditions) their
## volume-integral should be exactly (up to machine precision) the same.


###################
# # Tilted bottom boundary layer example
#
# This example is based on the similar [Oceananigans
# example](https://clima.github.io/OceananigansDocumentation/stable/generated/tilted_bottom_boundary_layer/)
# and simulates a two-dimensional oceanic bottom boundary layer in a domain that's tilted with
# respect to gravity. We simulate the perturbation away from a constant along-slope
# (y-direction) velocity constant density stratification.  This perturbation develops into a
# turbulent bottom boundary layer due to momentum loss at the bottom boundary.
# 
#
# First let's make sure we have all required packages installed.
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, Rasters, CairoMakie"
# ```
#
# ## Grid
#
# We start by creating a ``x, z`` grid with 64² cells and finer resolution near the bottom:

using Oceananigans
using Oceananigans.Units

Lx = 200meters
Lz = 100meters
Nx = 64
Nz = 64

refinement = 1.8 # controls spacing near surface (higher means finer spaced)
stretching = 10  # controls rate of stretching at bottom 

h(k) = (Nz + 1 - k) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
z_faces(k) = - Lz * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(topology = (Periodic, Flat, Bounded), size = (Nx, Nz),
                       x = (0, Lx), z = (0, Lz))

# Note that, with the `z` faces defined as above, the spacings near the bottom are approximately
# constant, becoming progressively coarser moving up.
#
# ## Tilting the domain
#
# We use a domain that's tilted with respect to gravity by

θ = 5; # degrees

# so that ``x`` is the along-slope direction, ``z`` is the across-sloce direction that
# is perpendicular to the bottom, and the unit vector anti-aligned with gravity is

ĝ = [sind(θ), 0, cosd(θ)]

# Changing the vertical direction impacts both the `gravity_unit_vector` for `Buoyancy` as well as
# the `rotation_axis` for Coriolis forces,

buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = -ĝ)

f₀ = 1e-4/second
coriolis = ConstantCartesianCoriolis(f = f₀, rotation_axis = ĝ)

# The tilting also affects the kind of density stratified flows we can model. The simulate an
# environment that's uniformly stratified, with a stratification frequency

N² = 1e-5/second^2;

# In a tilted coordinate, this can be achieved with

@inline constant_stratification(x, z, t, p) = p.N² * (x * p.ĝ[1] + z * p.ĝ[3]);

# However, this distribution is _not_ periodic in ``x`` and can't be explicitly modelled on an
# ``x``-periodic grid such as the one used here. Instead, we simulate periodic _perturbations_ away
# from the constant density stratification by imposing a constant stratification as a
# `BackgroundField`,

B_field = BackgroundField(constant_stratification, parameters=(; ĝ, N²))

# ## Bottom drag
#
# We impose bottom drag that follows Monin-Obukhov theory and include the background flow in the
# drag calculation, which is the only effect the background flow has on the problem

V∞ = 0.1meters/second
z₀ = 0.1meters # (roughness length)
κ = 0.4 # von Karman constant
z₁ = znodes(grid, Center())[1] # Closest grid center to the bottom
cᴰ = (κ / log(z₁ / z₀))^2 # Drag coefficient

@inline drag_u(x, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * u
@inline drag_v(x, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * (v + p.V∞)

drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)

# ## Create model and simulation
#
# We are now ready to create the model. We create a `NonhydrostaticModel` with an
# `UpwindBiasedFifthOrder` advection scheme, a `RungeKutta3` timestepper, and a constant viscosity
# and diffusivity.

closure = ScalarDiffusivity(ν=2e-4, κ=2e-4)

model = NonhydrostaticModel(; grid, buoyancy, coriolis, closure,
                            timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder(),
                            tracers = :b,
                            boundary_conditions = (u=u_bcs, v=v_bcs),
                            background_fields = (; b=B_field))

noise(x, z) = 1e-3 * randn() * exp(-(10z)^2/grid.Lz^2)
set!(model, u=noise, w=noise)

# The bottom-intensified noise above should accelerate the emergence of turbulence close to the
# wall.
#
# We are now ready to create the simulation. We begin by setting the initial time step
# conservatively, based on the smallest grid size of our domain and set-up a 

using Oceananigans.Units

simulation = Simulation(model, Δt = 0.5 * minimum_zspacing(grid) / V∞, stop_time = 12hours)

# We use `TimeStepWizard` to maximize Δt

wizard = TimeStepWizard(max_change=1.1, cfl=0.7)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

# ## Model diagnostics
#
# We set-up a custom progress messenger using `Oceanostics.ProgressMessengers`, which allows
# us to combine different `ProgressMessenger`s into one:

using Oceanostics.ProgressMessengers

walltime_per_timestep = StepDuration() # This needs to instantiated here, and not in the function below
progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false) + SimulationTime() + TimeStep() + MaxVelocities() + AdvectiveCFLNumber() + walltime_per_timestep)(simulation)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(400))

# We now define some useful diagnostics for the flow. Namely, we define `RichardsonNumber`,
# `RossbyNumber` and `ErtelPotentialVorticity`:

using Oceanostics

Ri = RichardsonNumber(model, add_background=true)
Ro = RossbyNumber(model)
PV = ErtelPotentialVorticity(model, add_background=true)

# Note that the calculation of these quantities depends on the alignment with the true (geophysical)
# vertical and the rotation axis. Oceanostics already takes that into consideration by using
# `model.buoyancy` and `model.coriolis`, making their calculation much easier. Furthermore, passing
# the flag `add_background=true` automatically adds the `model`'s `BackgroundField`s to the resolved
# perturbations, which is important in our case for the correct calculation of ``\nabla b`` with the
# background stratification.
#
# Now we write these quantities to a NetCDF file:

output_fields = (; Ri, Ro, PV, b = model.tracers.b + model.background_fields.tracers.b)

filename = "tilted_bottom_boundary_layer"
simulation.output_writers[:nc] = NetCDFOutputWriter(model, output_fields,
                                                    filename = joinpath(@__DIR__, filename),
                                                    schedule = TimeInterval(20minutes),
                                                    overwrite_existing = true)

# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# Now we'll read the results and plot an animation

using Rasters

ds = RasterStack(simulation.output_writers[:nc].filepath)

# We now use Makie to create the figure and its axes

using CairoMakie

set_theme!(Theme(fontsize = 20))
fig = Figure()

kwargs = (xlabel="x", ylabel="z", height=150, width=250)
ax1 = Axis(fig[2, 1]; title = "Ri", kwargs...)
ax2 = Axis(fig[2, 2]; title = "Ro", kwargs...)
ax3 = Axis(fig[2, 3]; title = "PV", kwargs...);

# Next we an `Observable` to lift the values at each specific time and plot
# heatmaps, along with their colorbars, with buoyancy contours on top

n = Observable(1)

bₙ = @lift set(ds.b[Ti=$n, yC=Near(0)], :xC => X, :zC => Z)

Riₙ = @lift set(ds.Ri[Ti=$n, yC=Near(0)], :xC => X, :zF => Z)
hm1 = heatmap!(ax1, Riₙ; colormap = :coolwarm, colorrange = (-1, +1))
contour!(ax1, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5)
Colorbar(fig[3, 1], hm1, vertical=false, height=8, ticklabelsize=14)
#
#Roₙ = @lift set(ds.Ro[Ti=$n, yF=Near(0)], :xF => X, :zF => Z)
#hm2 = heatmap!(ax2, Roₙ; colormap = :balance, colorrange = (-10, +10))
#contour!(ax2, bₙ; levels=10, color=:black, linestyle=:dash, linewidth=0.5)
#Colorbar(fig[3, 2], hm2, vertical=false, height=8, ticklabelsize=14)
#
#PVₙ = @lift set(ds.PV[Ti=$n, yF=Near(0)], :xF => X, :zF => Z)
#hm3 = heatmap!(ax3, PVₙ; colormap = :coolwarm, colorrange = N²*f₀.*(-1.5, +1.5))
#contour!(ax3, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5)
#Colorbar(fig[3, 3], hm3, vertical=false, height=8, ticklabelsize=14);
#
## Now we mark the time by placing a vertical line in the bottom panel and adding a helpful title
#
times = dims(ds, :Ti)
#title = @lift "Time = " * string(prettytime(times[$n]))
#fig[1, 1:3] = Label(fig, title, fontsize=24, tellwidth=false);
#
## Finally, we adjust the figure dimensions to fit all the panels and record a movie
#
#resize_to_layout!(fig)
#
@info "Animating..."
record(fig, filename * ".mp4", 1:length(times), framerate=10) do i
       n[] = i
end

# ![](tilted_bottom_boundary_layer.mp4)
#
# The animation shows negative PV being produced at the bottom due to drag, which leads to the
# emergence of centrifulgal-symmetric instabilities, which become turbulent and erode stratification
# (as can be seen by inspecting ``Ri``). Note that there are some boundary effects on the upper
# boundary, likely caused by interaction internal waves that are produced by the bottom turbulence.
# These effects are, to some degree, expected, and a sponge/relaxation layer at the top is needed to
# minimize them in a production-ready code.
