# # Eady baroclinic instability
#
# This example simulates the [Eady problem](https://en.wikipedia.org/wiki/Eady_model): the
# baroclinic instability of a uniformly stratified, uniformly sheared flow held in thermal-wind
# balance on an ``f``-plane. It is the canonical model for the baroclinic instability that fills the
# ocean and atmosphere with eddies. The setup follows the classic
# [Eady turbulence example](https://numericalearth.github.io/OceananigansMuseum/stable/generated/eady_turbulence/),
# scaled down to a submesoscale large-eddy simulation (LES) with a `DynamicSmagorinsky` closure, and
# uses Oceanostics to diagnose the energy conversion that powers the instability: available potential
# energy stored in the tilted background buoyancy field is released and converted into eddy kinetic
# energy.
#
# This is a three-dimensional LES, so it is heavier than the other examples, but it is sized to
# finish in a reasonable time. The resolution (`Nx, Ny, Nz`) and the `stop_time` below are the main
# cost knobs and can be reduced for a quicker look or increased for a more resolved run.
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie, NCDatasets"
# ```

# ## The background state
#
# We work in dimensional (SI) units. The domain is doubly periodic in the horizontal, with `x` the
# along-front direction and `y` the cross-front direction, and bounded in the vertical `z`. Following
# the classic Eady setup, the background state is parameterized by the Coriolis frequency `f`, the
# buoyancy frequency `N`, and the geostrophic shear `α = ∂U/∂z`. The background velocity and buoyancy
# are then
#
# ```math
# U(z) = α \left(z + \tfrac{H}{2}\right), \qquad B(y, z) = -α f \, y + N² z,
# ```
#
# which are in thermal-wind balance, ``f\,\partial_z U = -\partial_y B = α f``. We pick a weakly
# stratified, submesoscale-front regime so that the deformation radius stays small enough to resolve
# in three dimensions:

using Oceananigans
using Oceananigans.Units

f₀ = 1e-4      # [s⁻¹] Coriolis frequency
α  = 10 * f₀   # [s⁻¹] geostrophic shear ∂U/∂z
N  = 1e-3      # [s⁻¹] buoyancy frequency
H  = 50        # [m]   depth

coriolis = FPlane(f = f₀)

# From these we can form the cross-front buoyancy gradient `M² = α f`, the deformation radius
# `Lᵈ = N H / f`, and the wavelength of the fastest-growing Eady mode `λ ≈ 3.9 Lᵈ`, which sets a
# natural horizontal size for the domain:

M² = α * f₀              # cross-front buoyancy gradient ∂B/∂y (magnitude)
Ld = N * H / f₀          # deformation radius
λ  = 3.9 * Ld            # fastest-growing Eady wavelength

@info "Deformation radius Lᵈ ≈ $(round(Ld)) m, fastest Eady wavelength λ ≈ $(round(λ)) m"

# ## Grid
#
# We build a doubly-periodic grid one Eady wavelength wide, with a stretched vertical coordinate that
# is fine near the surface (where the submesoscale dynamics concentrate) and coarsens toward the
# bottom. The stretching follows the standard Oceananigans reference-to-stretched mapping: a uniform
# reference coordinate `h ∈ [0, 1]` is passed through a refinement/stretching generator that clusters
# grid faces near the surface.

Lx = Ly = 2000meters
Nx = Ny = 64
Nz = 32

refinement = 1.2   # controls spacing near the surface (higher means finer near the surface)
stretching = 5.0   # controls how quickly the spacing coarsens toward the bottom

## Normalized reference height, 0 at the bottom and 1 at the surface
h(k) = (k - 1) / Nz

## Linear near-surface generator
ζ₀(k) = 1 + (h(k) - 1) / refinement

## Bottom-intensified stretching function
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

## Generating function: z_faces(1) = -H at the bottom, z_faces(Nz+1) = 0 at the surface
z_faces(k) = H * (ζ₀(k) * Σ(k) - 1)

grid = RectilinearGrid(topology = (Periodic, Periodic, Bounded), size = (Nx, Ny, Nz),
                       x = (0, Lx), y = (0, Ly), z = z_faces)

# ## Background fields
#
# The cross-front buoyancy gradient `-α f y` is not periodic in `y`, so, just like the constant
# stratification in the [Tilted bottom boundary layer example](@ref), we cannot set it directly on
# the periodic grid. Instead we impose the full background buoyancy and the geostrophic shear as
# `BackgroundField`s and evolve only the periodic _perturbations_ away from them.

@inline U_background(x, y, z, t, p) = p.α * (z + p.H / 2)
@inline B_background(x, y, z, t, p) = - p.α * p.f * y + p.N^2 * z

background_parameters = (; α, f = f₀, N, H)
U_field = BackgroundField(U_background, parameters=background_parameters)
B_field = BackgroundField(B_background, parameters=background_parameters)

# ## Closure and model
#
# We keep the LES closure of the original example: a `DynamicSmagorinsky`, which computes the
# Smagorinsky coefficient dynamically from the resolved flow rather than fixing it a priori (here
# with its default `LagrangianAveraging`). We assemble a `NonhydrostaticModel` with a `WENO`
# advection scheme, the buoyancy `b` as the active tracer, and the background fields defined above.

closure = DynamicSmagorinsky()

model = NonhydrostaticModel(grid; coriolis, closure,
                            timestepper = :RungeKutta3,
                            advection = WENO(order=5),
                            buoyancy = BuoyancyTracer(), tracers = :b,
                            background_fields = (; u = U_field, b = B_field))

# ## Initial condition
#
# We seed the instability with small-amplitude random noise, damped toward the top and bottom
# boundaries so the perturbation projects onto interior modes, and then remove any net horizontal-mean
# velocity that the noise introduces. We use a fixed seed for reproducibility.

using Random
using Statistics: mean
Random.seed!(43)

Ξ(z) = randn() * (z / H) * (z / H + 1) # random noise that vanishes at z = 0 and z = -H

Ũ = 1e-1 * α * H    # velocity-noise amplitude
B̃ = 1e-2 * α * f₀   # buoyancy-noise amplitude

uᵢ(x, y, z) = Ũ * Ξ(z)
bᵢ(x, y, z) = B̃ * Ξ(z)

set!(model, u=uᵢ, v=uᵢ, b=bᵢ)

parent(model.velocities.u) .-= mean(interior(model.velocities.u))
parent(model.velocities.v) .-= mean(interior(model.velocities.v))

# ## Simulation
#
# We start from a conservative time step and let a `TimeStepWizard` adapt it as the eddies spin up.
# Two subtleties matter here. First, the wizard's advective CFL only sees the resolved (perturbation)
# velocities, so we cap the step ourselves from the background advective CFL, using the along-front
# grid spacing and the peak geostrophic velocity. Second, the `DynamicSmagorinsky` eddy viscosity
# grows sharply in the thin near-surface cells once the eddies saturate, so we also enforce a
# `diffusive_cfl`; without it the vertical diffusion would go unstable at these step sizes.

Ū = α * H                                     # peak background (geostrophic) velocity
max_Δt = 0.2 * minimum_xspacing(grid) / Ū     # keep the background advective CFL small
simulation = Simulation(model, Δt = 0.1 * max_Δt, stop_time = 6days)

wizard = TimeStepWizard(cfl=0.7, diffusive_cfl=0.5, max_change=1.1, max_Δt=max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# We report progress with a custom messenger built from `Oceanostics.ProgressMessengers`:

using Oceanostics.ProgressMessengers

walltime_per_timestep = StepDuration() # This needs to be instantiated here, and not in the function below
progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false) + SimulationTime() + TimeStep() + MaxVelocities() + AdvectiveCFLNumber() + walltime_per_timestep)(simulation)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# ## Diagnostics
#
# The signature of baroclinic instability is the release of available potential energy into eddy
# kinetic energy. We track that conversion with three Oceanostics diagnostics, all computed from the
# resolved _perturbation_ fields (the background fields are excluded, since `model.velocities` and
# `model.tracers.b` hold only the perturbations):
#
# - the eddy kinetic energy `KineticEnergy`,
# - the buoyancy production `BuoyancyProduction`, i.e. the vertical buoyancy flux ``w'b'`` that
#   converts potential to kinetic energy and appears as a source in the kinetic-energy budget,
# - and the perturbation potential energy `PotentialEnergy` ``= -b'z``, which reflects the available
#   potential energy released as the front slumps.
#
# We volume-integrate each one so we can follow the domain-wide energy budget in time:

using Oceanostics.KineticEnergyEquation: KineticEnergy, BuoyancyProduction
using Oceanostics.PotentialEnergyEquation: PotentialEnergy

∫KE = Integral(KineticEnergy(model))
∫wb = Integral(BuoyancyProduction(model))
∫PE = Integral(PotentialEnergy(model))

# For visualization we also keep two surface fields: the perturbation buoyancy `b'` and the vertical
# vorticity `ζ = ∂v/∂x - ∂u/∂y`, which highlights the developing eddies and filaments.

b′ = model.tracers.b
ζ  = ∂x(model.velocities.v) - ∂y(model.velocities.u)

# We write the volume integrals and the surface fields to two separate NetCDF files, since the
# surface fields need a horizontal slice (`indices`) that does not apply to the scalar integrals:

using NCDatasets

simulation.output_writers[:energetics] =
    NetCDFWriter(model, (; KE=∫KE, PE=∫PE, wb=∫wb),
                 filename = joinpath(@__DIR__, "eady_energetics.nc"),
                 schedule = TimeInterval(1hour),
                 overwrite_existing = true)

simulation.output_writers[:surface] =
    NetCDFWriter(model, (; b=b′, ζ),
                 filename = joinpath(@__DIR__, "eady_surface.nc"),
                 schedule = TimeInterval(2hours),
                 indices = (:, :, grid.Nz),
                 overwrite_existing = true)

# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# We now read the results back with NCDatasets:

ds_s = NCDataset(simulation.output_writers[:surface].filepath)
ds_e = NCDataset(simulation.output_writers[:energetics].filepath)

times   = ds_s["time"][:] # surface-snapshot times, used for the animation frames
times_e = ds_e["time"][:] # energetics times, sampled more frequently

x_caa = ds_s["x_caa"][:]; y_aca = ds_s["y_aca"][:] # buoyancy at (Center, Center)
x_faa = ds_s["x_faa"][:]; y_afa = ds_s["y_afa"][:] # vorticity at (Face, Face)

KE = ds_e["KE"][:]
PE = ds_e["PE"][:]
wb = ds_e["wb"][:]

# We build a figure with the two surface fields on top and the energy budget below:

using CairoMakie

set_theme!(Theme(fontsize = 18))
fig = Figure(size = (900, 850))

n = Observable(1)

## Surface perturbation buoyancy (the retained singleton z-dimension needs the extra index)
axb = Axis(fig[2, 1]; title = "surface b′", xlabel="x [m]", ylabel="y [m]", aspect=1)
bₙ = @lift ds_s["b"][:, :, 1, $n]
blim = @lift maximum(abs, ds_s["b"][:, :, 1, $n]) + eps()
hmb = heatmap!(axb, x_caa, y_aca, bₙ; colormap = :balance, colorrange = @lift((-$blim, $blim)))
Colorbar(fig[2, 2], hmb)

## Surface Rossby number ζ/f
axζ = Axis(fig[2, 3]; title = "surface ζ / f", xlabel="x [m]", ylabel="y [m]", aspect=1)
ζₙ = @lift ds_s["ζ"][:, :, 1, $n] ./ f₀
hmζ = heatmap!(axζ, x_faa, y_afa, ζₙ; colormap = :curl, colorrange = (-3, +3))
Colorbar(fig[2, 4], hmζ)

## Energy budget time series
axKE = Axis(fig[3, 1:4]; xlabel="time [days]", ylabel="∫KE, ∫PE − ∫PE₀ [m⁵ s⁻²]")
lines!(axKE, times_e ./ day, KE,           label = "∫KE dV (eddy kinetic energy)")
lines!(axKE, times_e ./ day, PE .- PE[1],  label = "∫PE dV − ∫PE₀ (released potential energy)")
axislegend(axKE, position=:lt, labelsize=13)

axwb = Axis(fig[4, 1:4]; xlabel="time [days]", ylabel="∫w′b′ dV [m⁵ s⁻³]")
lines!(axwb, times_e ./ day, wb, color=:purple, label = "∫w′b′ dV (buoyancy production)")
axislegend(axwb, position=:lt, labelsize=13)

## Moving time markers
for ax in (axKE, axwb)
    vlines!(ax, @lift(times[$n] / day), color=:black, linestyle=:dash)
end

title = @lift "Eady baroclinic instability, t = " * string(prettytime(times[$n]))
fig[1, 1:4] = Label(fig, title, fontsize=22, tellwidth=false)

# Finally we record the movie:

@info "Animating..."
record(fig, "eady_baroclinic_instability.mp4", 1:length(times), framerate=12) do i
    n[] = i
end

close(ds_s)
close(ds_e)

# ![](eady_baroclinic_instability.mp4)
#
# The surface panels show the front breaking up into a train of submesoscale eddies and filaments,
# with vertical vorticity reaching order-`f` values (Rossby numbers of order one) as the instability
# saturates. The lower panels tell the energetic story: the perturbation potential energy is drawn
# down (the front slumps and restratifies) while the eddy kinetic energy grows, and the buoyancy
# production `∫w'b' dV` stays positive throughout. Buoyant fluid rises and dense fluid sinks, which
# is exactly the conversion of available potential energy into eddy kinetic energy that defines
# baroclinic instability.
