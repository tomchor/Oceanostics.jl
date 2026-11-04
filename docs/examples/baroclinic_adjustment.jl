# # [Baroclinic adjustment and the energy budgets with a free surface](@id baroclinic_adjustment_example)
#
# In this example we spin up a pair of mesoscale fronts in a doubly-periodic channel of a
# `HydrostaticFreeSurfaceModel`, let them go baroclinically unstable, and close the volume-integrated
# potential and kinetic energy budgets. The model has a free surface, which exchanges a little energy
# with the flow through the pressure; the kinetic energy budget below carries that exchange in its
# pressure term.
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie, NCDatasets"
# ```

using Oceananigans
using Oceananigans.Units

# ## Parameters
#
# A 500 km by 1000 km channel, 1 km deep and uniformly stratified. The two fronts are 100 km wide and
# carry a cross-front buoyancy gradient `M²`, which we set through the balanced Richardson number
# `Ri = N²f²/M⁴` of the thermal wind inside them:

Lx = 500kilometers
Ly = 1000kilometers
H  = 1kilometer

f       = 1e-4            # [s⁻¹] Coriolis frequency
N²      = 1e-5            # [s⁻²] stratification
Ri      = 20              # []    balanced Richardson number inside each front
w_front = 100kilometers   # [m]   width of each front

M² = √N² * f / √Ri        # [s⁻²] cross-front buoyancy gradient inside each front

# ## Grid
#
# Doubly periodic in the horizontal and bounded in the vertical, with 10 km cells and 20 levels.

grid = RectilinearGrid(size = (50, 100, 20),
                       x = (0, Lx), y = (-Ly/2, Ly/2), z = (-H, 0),
                       topology = (Periodic, Periodic, Bounded))

# ## Derived quantities
#
# The deformation radius `Ld = NH/f` sets the scale of the instability: the fastest-growing wavelength
# is `λ ≈ 3.9 Ld` and the growth rate `σ ≈ 0.31 f/√Ri`. The free surface has a scale of its own, the
# external deformation radius `L_ext = √(gH)/f`: the energy stored in a geostrophic displacement of
# scale `L` is a fraction `(L/L_ext)²` of the kinetic energy of the depth-mean flow that balances it.
# For the eddies, whose scale is `Ld`, that fraction is `N²H/g`, a tenth of a percent here, so the
# surface is a small reservoir.

α     = M² / f                 # [s⁻¹]   thermal-wind shear inside a front
Ld    = √N² * H / f            # [m]     deformation radius
λ     = 3.9 * Ld               # [m]     fastest-growing wavelength
σ     = 0.31 * f / √Ri         # [s⁻¹]   growth rate
Δb    = M² * w_front           # [m s⁻²] buoyancy jump across each front
Ū     = α * H                  # [m s⁻¹] thermal-wind velocity at the surface
g     = Oceananigans.defaults.gravitational_acceleration
L_ext = √(g * H) / f           # [m]     external deformation radius

@info "Ld = $(round(Ld/1e3, digits=1)) km, λ = $(round(λ/1e3, digits=1)) km in a $(round(Int, Lx/1e3)) km channel, growth time 1/σ = $(prettytime(1/σ))"
@info "Δb = $(round(Δb, sigdigits=3)) m s⁻², Ū = $(round(Ū, digits=2)) m/s, L_ext = $(round(L_ext/1e3, digits=0)) km"

# ## Closure
#
# As a turbulence closure, we use an anisotropic diffusivity of momentum and buoyancy. The large
# horizontal diffusivity keeps the 10 km cells smooth under the centered advection scheme, and the
# vertical diffusivity is as small as we can make it while still resolving the flow fluctuations:

νₕ = κₕ = 300     # [m² s⁻¹]
ν_z = κ_z = 1e-3  # [m² s⁻¹]

closure = (HorizontalScalarDiffusivity(ν = νₕ, κ = κₕ), VerticalScalarDiffusivity(ν = ν_z, κ = κ_z))

# ## Model
#
# A `HydrostaticFreeSurfaceModel` with a `SplitExplicitFreeSurface`, which steps the barotropic mode in
# substeps short enough for the 100 m/s external gravity wave, at a Courant number of 0.7, so that the
# time step can follow the flow. The advection schemes are `Centered`, which add no dissipation of their
# own, so every sink in the budgets below is one we write down and compute.

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = FPlane(; f),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7),
                                    timestepper = :SplitRungeKutta3,
                                    momentum_advection = Centered(order=4),
                                    tracer_advection = Centered(order=4),
                                    closure)

# ## Initial condition
#
# The buoyancy is the stratification plus two ramps of width `w_front`, one rising at `y = -Ly/4` and
# one falling at `y = +Ly/4`. Their difference returns to zero at the edges of the domain, so `b` matches
# across the periodic boundary:

ramp(y, w) = min(max(0, y/w + 1/2), 1)

y₁ = -Ly/4
y₂ = +Ly/4

double_ramp(y, w) = ramp(y - y₁, w) - ramp(y - y₂, w)

# The velocity starts in thermal-wind balance with those fronts, referenced to the bottom, and a little
# noise on the buoyancy seeds the instability:

using Random
Random.seed!(8675309)

ϵb = 1e-2 * Δb    # noise amplitude, a percent of the front's buoyancy jump

bᵢ(x, y, z) = N² * z + Δb * double_ramp(y, w_front) + ϵb * randn()

ramp_prime(y, w) = (-w/2 < y < w/2) ? 1/w : zero(y)
double_ramp_prime(y, w) = ramp_prime(y - y₁, w) - ramp_prime(y - y₂, w)

uᵢ(x, y, z) = -(Δb * double_ramp_prime(y, w_front) / f) * (z + H)

# The surface starts in balance as well, which spares the run a barotropic adjustment in its first
# steps. The velocity is referenced to the bottom, where it vanishes, so the pressure gradient vanishes
# there too: the slope of the surface cancels the gradient of the hydrostatic anomaly at the bottom,
# `g ∂η/∂y = H ∂b/∂y` across each front. That is twice the slope that balances the depth-mean jet
# alone, since the depth-mean gradient of the anomaly, zero at the surface and largest at the bottom,
# adds as much again. Oceananigans integrates the anomaly over a full cell at the top rather than the
# half cell above the top centre, which shifts every column's pressure by half a cell of surface
# buoyancy, so the surface that balances the discrete model has `H + Δz/2` in place of `H`:

Δz = H / grid.Nz

ηᵢ(x, y, z) = Δb * (H + Δz/2) / g * (double_ramp(y, w_front) - 1/2)

set!(model, u = uᵢ, b = bᵢ, η = ηᵢ)
nothing #hide

# ## Simulation
#
# A `TimeStepWizard`, set up by `conjure_time_step_wizard!`, adjusts the step every ten iterations to keep
# the advective Courant number at 0.25, which is about an hour while the initial jets are the fastest flow,
# and never lets it exceed that hour, well inside the inertial period. The external gravity waves are left
# to the free surface's substeps:

simulation = Simulation(model, Δt = 1hour, stop_time = 40days)
conjure_time_step_wizard!(simulation, cfl = 0.7, max_Δt = 1hour)

using Oceanostics
add_callback!(simulation, ProgressMessengers.TimedMessenger(), IterationInterval(200))

# ## The potential energy budget
#
# Given `eₚ = -bz`, we multiply the budget equation for `b` by `-z`. Pulling `z` inside each derivative
# splits terms into a transport and a conversion, and over this domain every transport integrates
# away, leaving just the two conversions:
#
# ```math
# \frac{d}{dt}\int e_p\, \mathrm{d}V = -\int wb\, \mathrm{d}V + \int \Phi\, \mathrm{d}V ,
# ```
#
# with ``wb``
# ([`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion))
# the exchange with the kinetic energy and ``\Phi = \kappa\,\partial b/\partial z``
# ([`PotentialEnergyDiffusiveVerticalBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveVerticalBuoyancyFlux))
# the work diffusion does against gravity:

eₚ = PotentialEnergy(model)
wb = PotentialToKineticEnergyConversion(model)
Φ  = PotentialEnergyDiffusiveVerticalBuoyancyFlux(model)

∫eₚ  = ∫dV(eₚ)
∫wb  = ∫dV(wb)
∫Φ   = ∫dV(Φ)

∂ₜ∫eₚ = TimeDerivative(∫eₚ)

# ## The kinetic energy budget
#
# This budget is the other side of the exchange, written with `eₖ = ½(u² + v²)`, the energy of the two
# prognostic velocities in a hydrostatic model. The pressure enters through
# [`KineticEnergyPressureRedistribution`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyPressureRedistribution),
# ``\texttt{PR} = u_j \partial_j (p' + g\eta)``, the work of the whole pressure: the hydrostatic anomaly
# ``p'`` that the buoyancy sets and the surface pressure ``g\eta`` of the free surface. It is contracted
# with all three velocity components, the diagnosed ``w`` included, so the budget reads much as it would
# on a nonhydrostatic model. Both the pressure redistribution and the advective terms stay, since the top
# of the domain (the surface) has a nonzero ``w``. So
#
# ```math
# \frac{d}{dt}\int e_k\, \mathrm{d}V = -\int \texttt{ADV}\, \mathrm{d}V - \int \texttt{PR}\, \mathrm{d}V
#                                    + \int wb\, \mathrm{d}V - \int \varepsilon_k\, \mathrm{d}V ,
# ```
#
# with ``\texttt{ADV} = u_i \partial_j (u_j u_i)``
# ([`KineticEnergyAdvection`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyAdvection)). We can
# calculate these quantities with:

eₖ  = KineticEnergy(model)
εₖ  = KineticEnergyDissipationRate(model)
ADV = KineticEnergyEquation.KineticEnergyAdvection(model)
PR  = KineticEnergyPressureRedistribution(model)

∫eₖ  = ∫dV(eₖ)
∫εₖ  = ∫dV(εₖ)
∫ADV = ∫dV(ADV)
∫PR  = ∫dV(PR)

∂ₜ∫eₖ = TimeDerivative(∫eₖ)

# For the movie we keep the surface vorticity, buoyancy and displacement and the mixed layer depth, in
# one file with the volume integrals. Each surface map is averaged over its single vertical level, the
# top level of cells for ``\zeta`` and ``b`` and the top face of the grid for ``\eta``, which leaves it
# two-dimensional, with no vertical dimension to share with the others. The mixed layer depth is
# two-dimensional already. We use the default criterion of
# [`MixedLayerDepth`](@ref Oceanostics.FlowDiagnostics.MixedLayerDepth), the depth at which the buoyancy
# has dropped by ``10^{-4} g`` below its surface value, which the initial stratification reaches about
# 120 m down:

u, v, w = model.velocities
b = model.tracers.b
ζ = ∂x(v) - ∂y(u)
ζₛ = Average(Field(ζ, indices = (:, :, grid.Nz)), dims=3)
bₛ = Average(view(b, :, :, grid.Nz), dims=3)
ηₛ = Average(model.free_surface.displacement, dims=3)
MLD = MixedLayerDepth(grid, model.buoyancy.formulation, (; b))

# ## Output
#
# One writer for the four maps and the integrals, on `TimeInterval(6hours)`.

using NCDatasets
filename = joinpath(@__DIR__, "baroclinic_adjustment")

simulation.output_writers[:output] =
    NetCDFWriter(model, (; ζ = ζₛ, b = bₛ, η = ηₛ, MLD, ∂ₜ∫eₚ, ∫Φ, ∂ₜ∫eₖ, ∫wb, ∫εₖ, ∫ADV, ∫PR),
                 filename = filename,
                 schedule = TimeInterval(6hours),
                 overwrite_files = true)

# ## Run the simulation

run!(simulation)

# ## Process the results

using CairoMakie

ds = NCDataset(simulation.output_writers[:output].filepath)
times = ds["time"][:]
x_faa = ds["x_faa"][:]; y_afa = ds["y_afa"][:]
x_caa = ds["x_caa"][:]; y_aca = ds["y_aca"][:]
ζ_arr = ds["ζ"][:, :, :]     # the maps have no vertical dimension: (x, y, time)
b_arr = ds["b"][:, :, :]
η_arr = ds["η"][:, :, :]
MLD_arr = ds["MLD"][:, :, :]

ζlim   = maximum(abs, ζ_arr)
ηlim   = 0.9 * maximum(abs, η_arr)
MLDlim = 0.9 * maximum(abs, filter(isfinite, MLD_arr))

# Every budget record carries both tendencies and every source term at the same time. The first record
# has no earlier state to difference against and is written as zero, so the budgets start from the
# second.

nb      = 2:length(times)
t_bud   = times[nb]
deₚdt   = ds["∂ₜ∫eₚ"][nb]
deₖdt   = ds["∂ₜ∫eₖ"][nb]
Φ_bud   = ds["∫Φ"][nb]
wb_bud  = ds["∫wb"][nb]
εₖ_bud  = ds["∫εₖ"][nb]
ADV_bud = ds["∫ADV"][nb]
PR_bud  = ds["∫PR"][nb]
close(ds)

using Statistics: mean

# Every budget in sum-to-zero form: each curve is plotted with the sign it carries here, so each panel
# below adds up to its residual.

eₚ_resid = @. -deₚdt - wb_bud + Φ_bud
eₖ_resid = @. -deₖdt - ADV_bud - PR_bud + wb_bud - εₖ_bud

using Test                                                                            #hide
rms(x) = √(sum(abs2, x) / length(x))                                                  #hide
eₚ_terms = (deₚdt, wb_bud, Φ_bud)                                                     #hide
eₖ_terms = (deₖdt, ADV_bud, PR_bud, wb_bud, εₖ_bud)                                   #hide
## Each budget closes to a small fraction of its own largest term (measured 0.3% and 0.4%).          #hide
@test rms(eₚ_resid) < 0.015 * maximum(rms, eₚ_terms)                                  #hide
@test rms(eₖ_resid) < 0.015 * maximum(rms, eₖ_terms)                                  #hide
## `wb` is the same term in both, so it cancels from their sum                        #hide
@test rms(eₚ_resid .+ eₖ_resid) < 0.02 * rms(wb_bud)                                  #hide
## Integrated, the pressure work is only the exchange with the free surface, a fraction of a percent  #hide
## of the conversion (measured 0.2%).                                                                 #hide
@test mean(abs, PR_bud) < 0.01 * mean(abs, wb_bud)                                    #hide
## The adjustment does what it should: the fronts slump and the potential energy they release turns  #hide
## into kinetic energy, so the conversion is positive on average and leads the kinetic energy budget.#hide
@test mean(wb_bud) > 0                                                                #hide
@test rms(wb_bud) > 0.5 * rms(deₖdt)                                                  #hide
## The closure dissipates everywhere, with a non-negative integral in every record.                   #hide
@test all(εₖ_bud .≥ 0);                                                               #hide

# ## Plotting

set_theme!(Theme(fontsize = 18))
fig = Figure(size = (1000, 1750))

n = Observable(1)

panel_kwargs = (xlabel = "x [km]", ylabel = "y [km]", aspect = DataAspect())
axζ = Axis(fig[2, 1][1, 1]; title = "surface vorticity, ζ", panel_kwargs...)
axb = Axis(fig[2, 2][1, 1]; title = "surface buoyancy, b",  panel_kwargs...)
axη = Axis(fig[3, 1][1, 1]; title = "free surface, η",      panel_kwargs...)
axh = Axis(fig[3, 2][1, 1]; title = "mixed layer depth",    panel_kwargs...)

ζₙ = @lift ζ_arr[:, :, $n]
bₙ = @lift b_arr[:, :, $n]
ηₙ = @lift η_arr[:, :, $n]
MLDₙ = @lift MLD_arr[:, :, $n]

km = 1e-3
hmζ = heatmap!(axζ, x_faa .* km, y_afa .* km, ζₙ; colormap = :balance, colorrange = (-ζlim, ζlim))
Colorbar(fig[2, 1][1, 2], hmζ)

hmb = heatmap!(axb, x_caa .* km, y_aca .* km, bₙ; colormap = :thermal)
Colorbar(fig[2, 2][1, 2], hmb)

hmη = heatmap!(axη, x_caa .* km, y_aca .* km, ηₙ; colormap = :balance, colorrange = (-ηlim, ηlim))
Colorbar(fig[3, 1][1, 2], hmη)

hmh = heatmap!(axh, x_caa .* km, y_aca .* km, MLDₙ; colormap = :deep, colorrange = (-MLDlim, 0))
Colorbar(fig[3, 2][1, 2], hmh)

budget_kwargs = (xlabel = "time [days]", ylabel = "[m⁵ s⁻³]")

ax_p = Axis(fig[4, 1:2]; title = "Volume-integrated potential energy budget", budget_kwargs...)
lines!(ax_p, t_bud ./ day, -deₚdt,    label = "-d(∫eₚ)/dt")
lines!(ax_p, t_bud ./ day, -wb_bud,   label = "-∫wb dV")
lines!(ax_p, t_bud ./ day,  Φ_bud,    label = "∫Φ dV")
lines!(ax_p, t_bud ./ day,  eₚ_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_p; position = :rt, labelsize = 16, nbanks = 2)

ax_k = Axis(fig[5, 1:2]; title = "Volume-integrated kinetic energy budget", budget_kwargs...)
lines!(ax_k, t_bud ./ day, -deₖdt,    label = "-d(∫eₖ)/dt")
lines!(ax_k, t_bud ./ day, -ADV_bud,  label = "-∫ADV dV")
lines!(ax_k, t_bud ./ day,  wb_bud,   label = "∫wb dV")
lines!(ax_k, t_bud ./ day, -εₖ_bud,   label = "-∫εₖ dV")
lines!(ax_k, t_bud ./ day, -PR_bud,   label = "-∫PR dV (on the axis)")
lines!(ax_k, t_bud ./ day,  eₖ_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_k; position = :rt, labelsize = 16, nbanks = 2)

for ax in (ax_p, ax_k)
    vlines!(ax, @lift(times[$n] / day), color = :black, linestyle = :dot)
end

title = @lift "Baroclinic adjustment, t = " * prettytime(times[$n])
fig[1, 1:2] = Label(fig, title, fontsize = 22, tellwidth = false)

rowsize!(fig.layout, 1, Fixed(50))
rowsize!(fig.layout, 2, Fixed(500))
rowsize!(fig.layout, 3, Fixed(500))
rowsize!(fig.layout, 4, Fixed(300))
rowsize!(fig.layout, 5, Fixed(300))
colsize!(fig.layout, 1, Fixed(450))
colsize!(fig.layout, 2, Fixed(450))
resize_to_layout!(fig)

@info "Animating..."
record(fig, "baroclinic_adjustment.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
set_theme!() #hide
nothing #hide

# ![](baroclinic_adjustment.mp4)
