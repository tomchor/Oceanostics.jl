# # Eady baroclinic instability
#
# This example simulates the [Eady problem](https://en.wikipedia.org/wiki/Eady_model): the
# baroclinic instability of a uniformly stratified, uniformly sheared flow held in thermal-wind
# balance on an ``f``-plane. It is the canonical model for the baroclinic instability that fills the
# ocean and atmosphere with mesoscale and submesoscale eddies. Here we solve it in three dimensions
# as a large-eddy simulation (LES) with a `DynamicSmagorinsky` closure, and use Oceanostics to
# diagnose the energy conversion that powers the instability: available potential energy stored in
# the tilted background buoyancy field is released and converted into eddy kinetic energy.
#
# This is a heavier, three-dimensional example. It is meant to be run on its own (ideally on a GPU)
# rather than as part of the lightweight documentation build, and the resolution and run length
# below can be reduced for a quicker look.
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
# along-front direction and `y` the cross-front direction, and bounded in the vertical `z`. The
# background state consists of a uniform stratification `N²`, a uniform cross-front buoyancy gradient
# `M²`, and a geostrophic velocity `U(z)` in thermal-wind balance with them:
#
# ```math
# B(y, z) = N² z + M² y, \qquad f \, \frac{\partial U}{\partial z} = -\frac{\partial B}{\partial y} = -M².
# ```
#
# We pick a weakly stratified, submesoscale-front regime so that the deformation radius stays small
# enough to fit inside a domain we can resolve in three dimensions:

using Oceananigans
using Oceananigans.Units

f₀ = 1e-4     # [s⁻¹]   Coriolis frequency
N² = 1e-6     # [s⁻²]   background (vertical) buoyancy gradient
M² = 1e-7     # [s⁻²]   cross-front (horizontal) buoyancy gradient

# From these we can form the geostrophic shear `Λ = M²/f`, the deformation radius `Lᵈ = N H / f`,
# and the wavelength of the fastest-growing Eady mode `λ ≈ 3.9 Lᵈ`, which sets a natural horizontal
# size for the domain:

H  = 50        # [m] depth
Λ  = M² / f₀   # [s⁻¹] thermal-wind shear ∂U/∂z
Ld = sqrt(N²) * H / f₀       # deformation radius
λ  = 3.9 * Ld                # fastest-growing Eady wavelength

@info "Deformation radius Lᵈ ≈ $(round(Ld)) m, fastest Eady wavelength λ ≈ $(round(λ)) m"

# ## Grid
#
# We build a doubly-periodic grid one Eady wavelength wide, with a stretched vertical coordinate that
# is fine near the surface (where the submesoscale dynamics concentrate) and coarsens toward the
# bottom. The stretching follows the standard Oceananigans reference-to-stretched mapping: a uniform
# reference coordinate `h ∈ [0, 1]` is passed through a refinement/stretching generator that clusters
# grid faces near the surface.

Lx = Ly = 2000meters
Nx = Ny = 128
Nz = 48

refinement = 1.5   # controls spacing near the surface (higher means finer near the surface)
stretching = 6.0   # controls how quickly the spacing coarsens toward the bottom

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
# The cross-front buoyancy gradient `M² y` is not periodic in `y`, so, just like the constant
# stratification in the [Tilted bottom boundary layer example](@ref), we cannot set it directly on
# the periodic grid. Instead we impose the full background buoyancy and the geostrophic shear as
# `BackgroundField`s and evolve only the periodic _perturbations_ away from them. We center `U(z)` so
# that its vertical average vanishes, which keeps the advective time step small.

@inline B_background(x, y, z, t, p) = p.N² * z + p.M² * y
@inline U_background(x, y, z, t, p) = -p.Λ * (z + p.H / 2)

B_field = BackgroundField(B_background, parameters=(; N², M²))
U_field = BackgroundField(U_background, parameters=(; Λ, H))

# ## Coriolis and closure
#
# We use a traditional `FPlane` and a `DynamicSmagorinsky` closure. `DynamicSmagorinsky` computes the
# Smagorinsky coefficient dynamically from the resolved flow rather than fixing it a priori; here we
# use its default `LagrangianAveraging`, which averages the coefficient along Lagrangian trajectories
# and so makes no assumption of statistical homogeneity.

coriolis = FPlane(f = f₀)
closure = DynamicSmagorinsky()

# ## Model and initial condition
#
# We assemble a `NonhydrostaticModel` with a `WENO` advection scheme, the buoyancy `b` as the active
# tracer, and the background fields defined above. We then seed the flow with small-amplitude random
# noise on the velocities to trigger the instability, using a fixed seed for reproducibility.

model = NonhydrostaticModel(grid; coriolis, closure,
                            timestepper = :RungeKutta3,
                            advection = WENO(order=5),
                            buoyancy = BuoyancyTracer(), tracers = :b,
                            background_fields = (; u = U_field, b = B_field))

using Random
Random.seed!(43)

uᵢ(x, y, z) = 1e-3 * randn()
set!(model, u=uᵢ, v=uᵢ, w=uᵢ)

# ## Simulation
#
# We set the initial time step from the finest vertical spacing and the geostrophic velocity scale,
# and let a `TimeStepWizard` adapt it as the eddies spin up:

Umax = Λ * H / 2   # magnitude of the background surface velocity
simulation = Simulation(model, Δt = 0.2 * minimum_zspacing(grid) / Umax, stop_time = 8days)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=1minute)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

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
