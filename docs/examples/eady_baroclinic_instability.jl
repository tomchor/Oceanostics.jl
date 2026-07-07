# # Eady baroclinic instability
#
# This example simulates the [Eady problem](https://en.wikipedia.org/wiki/Eady_model): the
# baroclinic instability of a uniformly stratified, uniformly sheared flow held in thermal-wind
# balance on an ``f``-plane. It is a direct port of the classic
# [Eady turbulence example](https://numericalearth.github.io/OceananigansMuseum/v0.74.1/generated/eady_turbulence/)
# to up-to-date Oceananigans syntax, with the bottom drag removed.
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units

# ## Parameters
#
# We collect all the primitive parameters of the problem up front, before constructing anything.

## Domain size
Lx = Ly = 1e6   # [m]   horizontal extent (1000 km)
H  = 4e3        # [m]   depth

## Physical parameters
f = 1e-4        # [s⁻¹] Coriolis frequency
α = 10 * f      # [s⁻¹] geostrophic shear ∂U/∂z
N = 1e-3        # [s⁻¹] buoyancy frequency

# ## Derived dynamical quantities
#
# From those parameters we can form the quantities that characterize the flow: the cross-front and
# vertical buoyancy gradients, the deformation radius `Ld = N H / f`, the wavelength of the
# fastest-growing Eady mode `λ ≈ 3.9 Ld`, the balanced Richardson number `Ri = N²/α²`, the
# thermal-wind velocity scale `Ū = α H`, and the maximum Eady growth rate `σ ≈ 0.31 M²/N`.

M² = α * f          # [s⁻²]   cross-front buoyancy gradient |∂B/∂y|
N² = N^2            # [s⁻²]   vertical buoyancy gradient
Ld = N * H / f      # [m]     first baroclinic deformation radius
λ  = 3.9 * Ld       # [m]     wavelength of the fastest-growing Eady mode
Ri = N² / α^2       # []      balanced Richardson number
Ū  = α * H          # [m s⁻¹] peak background (thermal-wind) velocity
σ  = 0.31 * M² / N  # [s⁻¹]   maximum Eady growth rate

@info "Deformation radius Ld = $(round(Ld/1e3, digits=1)) km, fastest Eady wavelength λ = $(round(λ/1e3, digits=1)) km"
@info "Balanced Richardson number Ri = $(round(Ri, digits=2)), thermal-wind velocity Ū = $(round(Ū, digits=3)) m/s, Eady growth time 1/σ = $(prettytime(1/σ))"

# ## Grid
#
# We use a mesoscale-resolving grid, periodic in the horizontal and bounded in the vertical:

grid = RectilinearGrid(size = (48, 48, 16), extent = (Lx, Ly, H))

# ## Coriolis and background state
#
# The flow is set up on an ``f``-plane. The background velocity increases linearly with height, and
# the background buoyancy combines the geostrophic (cross-front) component with a stable
# stratification. They are in thermal-wind balance, ``f\,\partial_z U = -\partial_y B``:

coriolis = FPlane(f = f)

U(x, y, z, t, p) = + p.α * (z + p.H)
B(x, y, z, t, p) = - p.α * p.f * y + p.N^2 * z

background_parameters = (; α, f, N, H)
U_field = BackgroundField(U, parameters=background_parameters)
B_field = BackgroundField(B, parameters=background_parameters)

# ## Turbulence closures
#
# We dissipate variance with a Laplacian vertical diffusivity and a biharmonic horizontal
# diffusivity, applied simultaneously as a tuple of two closures:

Δx = minimum_xspacing(grid)
κ₂z = 1e-2                # [m² s⁻¹] Laplacian vertical viscosity and diffusivity
κ₄h = 1e-1 / day * Δx^4   # [m⁴ s⁻¹] biharmonic horizontal viscosity and diffusivity

vertical_diffusivity   = VerticalScalarDiffusivity(ν=κ₂z, κ=κ₂z)
biharmonic_diffusivity = HorizontalScalarBiharmonicDiffusivity(ν=κ₄h, κ=κ₄h)

# ## Model
#
# We build a `NonhydrostaticModel` with fifth-order `WENO` advection, a third-order Runge-Kutta
# timestepper, the buoyancy `b` as the active tracer, and the background fields and closures defined
# above. Following the request, there is no bottom drag, so the vertical boundaries are free-slip.

model = NonhydrostaticModel(grid;
                            advection = WENO(order=5),
                            timestepper = :RungeKutta3,
                            coriolis = coriolis,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            background_fields = (b=B_field, u=U_field),
                            closure = (vertical_diffusivity, biharmonic_diffusivity))

# ## Initial condition
#
# We seed the instability with small-amplitude random noise, damped toward the top and bottom
# boundaries so it projects onto interior modes, and then remove any net horizontal-mean velocity the
# noise introduces:

Ξ(z) = randn() * z / H * (z / H + 1) # noise that vanishes at z = 0 and z = -H

Ũ = 1e-1 * α * H   # velocity-noise amplitude
B̃ = 1e-2 * α * f   # buoyancy-noise amplitude

uᵢ(x, y, z) = Ũ * Ξ(z)
vᵢ(x, y, z) = Ũ * Ξ(z)
bᵢ(x, y, z) = B̃ * Ξ(z)

set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

using Statistics: mean
parent(model.velocities.u) .-= mean(interior(model.velocities.u))
parent(model.velocities.v) .-= mean(interior(model.velocities.v))

# ## Simulation
#
# The initial time step is set from the most restrictive of the advective and diffusive limits, and a
# `TimeStepWizard` adapts it as the eddies spin up:

max_Δt = min(Δx / Ū, Δx^4 / κ₄h, Δx^2 / κ₂z, 1 / N)

simulation = Simulation(model, Δt = max_Δt, stop_time = 16days)

wizard = TimeStepWizard(cfl=0.85, max_change=1.1, max_Δt=max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# We report progress with a simple messenger:

using Printf

start_time = time_ns()
progress(sim) = @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
                        sim.model.clock.iteration, prettytime(sim.model.clock.time),
                        prettytime(1e-9 * (time_ns() - start_time)), prettytime(sim.Δt),
                        AdvectiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# ## Output
#
# We save the vertical vorticity `ζ = ∂v/∂x - ∂u/∂y` and the horizontal divergence `δ = -∂w/∂z` every
# few hours:

u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)
δ = -∂z(w)

filename = joinpath(@__DIR__, "eady_baroclinic_instability.jld2")
simulation.output_writers[:fields] =
    JLD2Writer(model, (; ζ, δ),
               schedule = TimeInterval(4hours),
               filename = filename,
               overwrite_existing = true)

# ## Run the simulation and process results
#
# To run the simulation:

run!(simulation)

# We read the vorticity and divergence back as `FieldTimeSeries`:

using CairoMakie

ζ_timeseries = FieldTimeSeries(filename, "ζ")
δ_timeseries = FieldTimeSeries(filename, "δ")

times = ζ_timeseries.times
xζ, yζ, zζ = nodes(ζ_timeseries)
xδ, yδ, zδ = nodes(δ_timeseries)

k = grid.Nz # surface level

ζmax = maximum(abs, interior(ζ_timeseries))
δmax = maximum(abs, interior(δ_timeseries))

# We now build the figure and animate the surface fields over time:

set_theme!(Theme(fontsize = 18))
fig = Figure(size = (1100, 560))

n = Observable(1)

axζ = Axis(fig[2, 1]; title = "vertical vorticity, ζ", xlabel="x [km]", ylabel="y [km]", aspect=1)
axδ = Axis(fig[2, 3]; title = "horizontal divergence, δ", xlabel="x [km]", ylabel="y [km]", aspect=1)

ζₙ = @lift interior(ζ_timeseries[$n])[:, :, k]
δₙ = @lift interior(δ_timeseries[$n])[:, :, k]

hmζ = heatmap!(axζ, xζ ./ 1e3, yζ ./ 1e3, ζₙ; colormap = :balance, colorrange = (-ζmax, ζmax))
Colorbar(fig[2, 2], hmζ)

hmδ = heatmap!(axδ, xδ ./ 1e3, yδ ./ 1e3, δₙ; colormap = :balance, colorrange = (-δmax, δmax))
Colorbar(fig[2, 4], hmδ)

title = @lift "Eady turbulence, t = " * prettytime(times[$n])
fig[1, 1:4] = Label(fig, title, fontsize=22, tellwidth=false)

@info "Animating..."
record(fig, "eady_baroclinic_instability.mp4", 1:length(times), framerate=12) do i
    n[] = i
end

# ![](eady_baroclinic_instability.mp4)
#
# As the front becomes baroclinically unstable it rolls up into a field of mesoscale eddies, with the
# vertical vorticity organizing into cyclonic and anticyclonic patches and the horizontal divergence
# marking the frontogenetic regions between them.
