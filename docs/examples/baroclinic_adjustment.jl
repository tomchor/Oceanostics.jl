# # [Baroclinic adjustment and the energy budgets with a free surface](@id baroclinic_adjustment_example)
#
# In this example we spin up a pair of mesoscale fronts in a doubly-periodic channel of a
# `HydrostaticFreeSurfaceModel`, let them go baroclinically unstable, and close the volume-integrated
# potential and kinetic energy budgets. The model has a free surface, which stores energy of its own
# and exchanges it with the kinetic energy through the barotropic pressure gradient, so there are three
# budgets to close rather than two.
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
# Periodicity in `y` is what the double front buys us, and it is what makes the transport terms of every
# budget below integrate to zero. A single (Eady) front would need a uniform background buoyancy gradient,
# which makes the domain's potential energy infinite.

grid = RectilinearGrid(size = (50, 100, 20),
                       x = (0, Lx), y = (-Ly/2, Ly/2), z = (-H, 0),
                       topology = (Periodic, Periodic, Bounded))

# ## Derived quantities
#
# The deformation radius `Ld = NH/f` sets the scale of the instability: the fastest-growing wavelength
# is `λ ≈ 3.9 Ld` and the growth rate `σ ≈ 0.31 f/√Ri`. The free surface has a scale of its own, the
# external deformation radius `L_ext = √(gH)/f`: the energy stored in a geostrophic displacement of
# scale `L` is a fraction `(L/L_ext)²` of the kinetic energy of the depth-mean flow that balances it,
# which is why that reservoir is small, but not negligible, for fronts a few hundred kilometres wide.

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
# Laplacian diffusion of momentum and buoyancy, horizontal and vertical, both explicit in time, which
# is what lets the diagnostics below see every flux the model applies (see the budget pages for the
# vertically implicit case). The horizontal coefficient keeps the 10 km cells smooth under the centred
# advection scheme; the vertical one sits at the energetic end of what the mesoscale ocean mixes with,
# so that the work diffusion does against gravity shows in the potential energy budget.

νₕ = κₕ = 300     # [m² s⁻¹]
ν_z = κ_z = 1e-3  # [m² s⁻¹]

closure = (HorizontalScalarDiffusivity(ν = νₕ, κ = κₕ), VerticalScalarDiffusivity(ν = ν_z, κ = κ_z))

# ## Model
#
# A `HydrostaticFreeSurfaceModel` with its default free surface on this grid, an `ImplicitFreeSurface`,
# which steps the barotropic gravity waves implicitly and so lets the time step follow the flow rather
# than the 100 m/s external wave. The advection schemes are `Centered`, which add no dissipation of their
# own, so every sink in the budgets below is one we write down and compute.

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = FPlane(; f),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
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

# Referenced to the bottom, the thermal wind has a depth mean, `Ū/2` inside each front. In a
# hydrostatic model that jet is balanced by the slope of the free surface, `g ∂η/∂y = -f ū`, so we start
# the surface in balance as well, which spares the run a barotropic adjustment in its first steps:

ηᵢ(x, y, z) = Δb * H / 2g * (double_ramp(y, w_front) - 1/2)

set!(model, u = uᵢ, b = bᵢ, η = ηᵢ)
nothing #hide

# ## Simulation
#
# One hour is a quarter of the advective time step limit and well inside the inertial one, and the
# external gravity waves are implicit, so the step is fixed:

simulation = Simulation(model, Δt = 1hour, stop_time = 40days)

using Oceanostics
add_callback!(simulation, ProgressMessengers.TimedMessenger(), IterationInterval(100))

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
# the work diffusion does against gravity. The free surface does not enter: the surface flux of
# ``e_p`` carries a factor ``z = 0``, and ``\eta`` is not in the buoyancy equation.

eₚ = PotentialEnergy(model)
wb = PotentialToKineticEnergyConversion(model)
Φ  = PotentialEnergyDiffusiveVerticalBuoyancyFlux(model)

# ## The kinetic energy budget
#
# The other side of the exchange, written with `eₖ = ½(u² + v²)`, the energy of the two velocities a
# hydrostatic model steps. `wb` is the same term in both budgets and carries opposite signs, so it
# cancels from their sum. The free surface adds a term: the barotropic pressure gradient `g∇η` works on
# the flow, and
# [`KineticEnergyBarotropicPressureRedistribution`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyBarotropicPressureRedistribution)
# is that work, ``\texttt{BPR} = \boldsymbol{u}_h \cdot \nabla_h (g\eta)``, counted with the sign of the
# other pressure term. An `ImplicitFreeSurface` applies the gradient in its free-surface step, outside
# the tendency the model computes, so a budget written from the tendency's terms is short by exactly
# that term. The advective transport stays in as well: the top of the domain is open, since the ``w``
# the model diagnoses there is ``\partial_t \eta``, so a little ``e_k`` is advected through it, and the
# fourth-order centred scheme is not exactly energy-conserving; the two together are a percent of the
# conversion. So
#
# ```math
# \frac{d}{dt}\int e_k\, \mathrm{d}V = -\int \texttt{ADV}\, \mathrm{d}V + \int wb\, \mathrm{d}V
#                                    - \int \varepsilon_k\, \mathrm{d}V - \int \texttt{BPR}\, \mathrm{d}V ,
# ```
#
# with ``\texttt{ADV} = u_i \partial_j (u_j u_i)``
# ([`KineticEnergyAdvection`](@ref Oceanostics.KineticEnergyEquation.KineticEnergyAdvection)).
#
# Pointwise, the barotropic pressure is as hard at work as any term here: the depth-integrated
# ``\texttt{BPR}`` in the movie below is several times the conversion ``wb``. Integrated over the domain
# it is a fraction of a percent of it, because the geostrophic jets keep their transport, and with it
# the displacement of the surface, while the eddies grow. The work moves kinetic energy between columns
# far more than it stores it. We measure both below.

eₖ  = KineticEnergy(model)
εₖ  = KineticEnergyDissipationRate(model)
ADV = KineticEnergyEquation.KineticEnergyAdvection(model)
BPR = KineticEnergyBarotropicPressureRedistribution(model)

# ## The free surface's budget
#
# Where the kinetic energy loses that work, the free surface gains it. Its potential energy per unit
# area is `eₛ = ½gη²`
# ([`FreeSurfacePotentialEnergy`](@ref Oceanostics.PotentialEnergyEquation.FreeSurfacePotentialEnergy)),
# and the kinematic condition makes its tendency `gη ∂ₜη = gη w` at the surface
# ([`FreeSurfacePotentialEnergyTendency`](@ref Oceanostics.PotentialEnergyEquation.FreeSurfacePotentialEnergyTendency)).
# Integrated by parts across the periodic boundaries, the work of the barotropic pressure gradient is
# exactly that tendency,
#
# ```math
# \frac{d}{dt}\int e_s\, \mathrm{d}A = \int g\eta\, w|_{\mathrm{surface}}\, \mathrm{d}A
#                                    = \int \texttt{BPR}\, \mathrm{d}V ,
# ```
#
# so the three reservoirs close together, and the sum of the three budgets is the total energy's.

eₛ   = FreeSurfacePotentialEnergy(model)
∂ₜeₛ = FreeSurfacePotentialEnergyTendency(model)

∫eₚ  = ∫dV(eₚ)
∫Φ   = ∫dV(Φ)
∫eₖ  = ∫dV(eₖ)
∫wb  = ∫dV(wb)
∫εₖ  = ∫dV(εₖ)
∫ADV = ∫dV(ADV)
∫BPR = ∫dV(BPR)
∫absBPR = ∫dV(abs(BPR))
∫abswb  = ∫dV(abs(wb))
∫eₛ   = Integral(eₛ, dims=(1, 2))
∫∂ₜeₛ = Integral(∂ₜeₛ, dims=(1, 2))

∂ₜ∫eₚ = TimeDerivative(∫eₚ)
∂ₜ∫eₖ = TimeDerivative(∫eₖ)
∂ₜ∫eₛ = TimeDerivative(∫eₛ)

# For the movie we keep the surface vorticity and buoyancy, the free surface, and the barotropic
# pressure work integrated over the depth of each column, which shows how large the term is before the
# area integral cancels most of it. `η` lives on the top face of the grid rather than in a cell, so it is
# averaged over that one level, which turns it into a surface map the writer takes as it takes the depth
# integral:

u, v, w = model.velocities
b = model.tracers.b
ζ = ∂x(v) - ∂y(u)
η = Average(model.free_surface.displacement, dims=3)
∫BPRdz = Integral(BPR, dims=3)

# ## Output
#
# A *snapshot* writer for the surface slices of ``\zeta`` and ``b``, a *surface* writer for the two maps
# that are already two-dimensional, and a *budget* writer for the integrals, all on
# `TimeInterval(6hours)`.

using NCDatasets
filename = joinpath(@__DIR__, "baroclinic_adjustment")

simulation.output_writers[:fields] =
    NetCDFWriter(model, (; ζ, b),
                 filename = filename,
                 schedule = TimeInterval(6hours),
                 indices = (:, :, grid.Nz),
                 overwrite_files = true)

simulation.output_writers[:surface] =
    NetCDFWriter(model, (; η, ∫BPRdz),
                 filename = filename * "_surface",
                 schedule = TimeInterval(6hours),
                 overwrite_files = true)

simulation.output_writers[:budget] =
    NetCDFWriter(model, (; ∂ₜ∫eₚ, ∫Φ, ∂ₜ∫eₖ, ∫wb, ∫εₖ, ∫ADV, ∫BPR, ∂ₜ∫eₛ, ∫∂ₜeₛ, ∫absBPR, ∫abswb),
                 filename = filename * "_budget",
                 schedule = TimeInterval(6hours),
                 overwrite_files = true)

# ## Run the simulation

run!(simulation)

# ## Process the results

using CairoMakie

ds = NCDataset(simulation.output_writers[:fields].filepath)
times = ds["time"][:]
x_faa = ds["x_faa"][:]; y_afa = ds["y_afa"][:]
x_caa = ds["x_caa"][:]; y_aca = ds["y_aca"][:]
ζ_arr = ds["ζ"][:, :, 1, :]
b_arr = ds["b"][:, :, 1, :]
close(ds)

ds_s = NCDataset(simulation.output_writers[:surface].filepath)
η_arr = ds_s["η"][:, :, :]          # (Center, Center, Nothing)-located: no z dimension to slice
BPR_arr = ds_s["∫BPRdz"][:, :, :]
close(ds_s)

ζlim = maximum(abs, ζ_arr)
ηlim = maximum(abs, η_arr)
BPRlim = maximum(abs, BPR_arr)

# Every budget record carries both tendencies and every source term at the same time. The first record
# has no earlier state to difference against and is written as zero, so the budgets start from the
# second.

ds_b = NCDataset(simulation.output_writers[:budget].filepath)
nb   = 2:length(ds_b["time"])

t_bud   = ds_b["time"][nb]
deₚdt   = ds_b["∂ₜ∫eₚ"][nb]
deₖdt   = ds_b["∂ₜ∫eₖ"][nb]
deₛdt   = ds_b["∂ₜ∫eₛ"][nb]
Φ_bud   = ds_b["∫Φ"][nb]
wb_bud  = ds_b["∫wb"][nb]
εₖ_bud  = ds_b["∫εₖ"][nb]
ADV_bud = ds_b["∫ADV"][nb]
BPR_bud = ds_b["∫BPR"][nb]
gηw_bud = ds_b["∫∂ₜeₛ"][nb]
absBPR_bud = ds_b["∫absBPR"][nb]
abswb_bud  = ds_b["∫abswb"][nb]
close(ds_b)

# The two faces of the barotropic pressure work, pointwise and integrated, relative to the conversion:

using Statistics: mean
@info "∫|BPR| dV is $(round(mean(absBPR_bud ./ abswb_bud), digits=1)) times ∫|wb| dV on average, and ∫BPR dV is $(round(100 * mean(abs, BPR_bud) / mean(abs, wb_bud), digits=2))% of ∫wb dV"


# Every budget in sum-to-zero form: each curve is plotted with the sign it carries here, so each panel
# below adds up to its residual.

eₚ_resid = @. -deₚdt - wb_bud + Φ_bud
eₖ_resid = @. -deₖdt - ADV_bud + wb_bud - εₖ_bud - BPR_bud
eₛ_resid = @. -deₛdt + BPR_bud

using Test                                                                            #hide
rms(x) = √(sum(abs2, x) / length(x))                                                  #hide
eₚ_terms = (deₚdt, wb_bud, Φ_bud)                                                     #hide
eₖ_terms = (deₖdt, ADV_bud, wb_bud, εₖ_bud, BPR_bud)                                  #hide
## Each budget closes to a small fraction of its own largest term (measured 0.3%, 0.2% and 4%).      #hide
@test rms(eₚ_resid) < 0.015 * maximum(rms, eₚ_terms)                                  #hide
@test rms(eₖ_resid) < 0.015 * maximum(rms, eₖ_terms)                                  #hide
@test rms(eₛ_resid) < 0.15 * rms(BPR_bud)                                             #hide
## `wb` is the same term in both, so it cancels from their sum                        #hide
@test rms(eₚ_resid .+ eₖ_resid) < 0.02 * rms(wb_bud)                                  #hide
## The work of the barotropic pressure gradient is the free surface's energy tendency, exactly, at    #hide
## every output time; pointwise it is larger than the conversion and integrated it is a fraction of   #hide
## a percent of it (measured 6 times and 0.2%).                                                       #hide
@test gηw_bud ≈ BPR_bud rtol=1e-8                                                     #hide
@test mean(absBPR_bud ./ abswb_bud) > 1                                               #hide
@test mean(abs, BPR_bud) < 0.01 * mean(abs, wb_bud)                                   #hide
## The adjustment does what it should: the fronts slump and the potential energy they release turns  #hide
## into kinetic energy, so the conversion is positive on average and leads the kinetic energy budget.#hide
@test mean(wb_bud) > 0                                                                #hide
@test rms(wb_bud) > 0.5 * rms(deₖdt)                                                  #hide
## The closure dissipates everywhere, with a non-negative integral in every record.                   #hide
@test all(εₖ_bud .≥ 0);                                                               #hide

# ## Plotting

set_theme!(Theme(fontsize = 18))
fig = Figure(size = (1000, 2050))

n = Observable(1)

panel_kwargs = (xlabel = "x [km]", ylabel = "y [km]", aspect = DataAspect())
axζ = Axis(fig[2, 1][1, 1]; title = "surface vorticity, ζ",         panel_kwargs...)
axb = Axis(fig[2, 2][1, 1]; title = "surface buoyancy, b",          panel_kwargs...)
axη = Axis(fig[3, 1][1, 1]; title = "free surface, η",              panel_kwargs...)
axp = Axis(fig[3, 2][1, 1]; title = "barotropic pressure work, ∫BPR dz", panel_kwargs...)

ζₙ = @lift ζ_arr[:, :, $n]
bₙ = @lift b_arr[:, :, $n]
ηₙ = @lift η_arr[:, :, $n]
BPRₙ = @lift BPR_arr[:, :, $n]

km = 1e-3
hmζ = heatmap!(axζ, x_faa .* km, y_afa .* km, ζₙ; colormap = :balance, colorrange = (-ζlim, ζlim))
Colorbar(fig[2, 1][1, 2], hmζ)

hmb = heatmap!(axb, x_caa .* km, y_aca .* km, bₙ; colormap = :thermal)
Colorbar(fig[2, 2][1, 2], hmb)

hmη = heatmap!(axη, x_caa .* km, y_aca .* km, ηₙ; colormap = :balance, colorrange = (-ηlim, ηlim))
Colorbar(fig[3, 1][1, 2], hmη)

hmp = heatmap!(axp, x_caa .* km, y_aca .* km, BPRₙ; colormap = :balance, colorrange = (-BPRlim, BPRlim))
Colorbar(fig[3, 2][1, 2], hmp)

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
lines!(ax_k, t_bud ./ day, -BPR_bud,  label = "-∫BPR dV (on the axis)")
lines!(ax_k, t_bud ./ day,  eₖ_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_k; position = :rt, labelsize = 16, nbanks = 2)

## The first record holds the surface's adjustment to the discretized balance, a burst an order of
## magnitude above everything that follows, which is kept off this axis.
ax_s = Axis(fig[6, 1:2]; title = "Area-integrated free surface potential energy budget", budget_kwargs...)
lines!(ax_s, t_bud ./ day, -deₛdt,    label = "-d(∫eₛ)/dt")
lines!(ax_s, t_bud ./ day,  BPR_bud,  label = "∫BPR dV")
lines!(ax_s, t_bud ./ day,  eₛ_resid, label = "residual", color = :black, linestyle = :dash)
ylims!(ax_s, 1.5 .* extrema(BPR_bud[t_bud .> 1day]))
axislegend(ax_s; position = :rt, labelsize = 16, nbanks = 2)

for ax in (ax_p, ax_k, ax_s)
    vlines!(ax, @lift(times[$n] / day), color = :black, linestyle = :dot)
end

title = @lift "Baroclinic adjustment, t = " * prettytime(times[$n])
fig[1, 1:2] = Label(fig, title, fontsize = 22, tellwidth = false)

rowsize!(fig.layout, 1, Fixed(50))
rowsize!(fig.layout, 2, Fixed(500))
rowsize!(fig.layout, 3, Fixed(500))
rowsize!(fig.layout, 4, Fixed(300))
rowsize!(fig.layout, 5, Fixed(300))
rowsize!(fig.layout, 6, Fixed(300))
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
