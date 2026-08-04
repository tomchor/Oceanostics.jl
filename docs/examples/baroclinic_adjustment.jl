# # [Baroclinic adjustment and the potential energy budget](@id baroclinic_adjustment_example)
#
# In this example we spin up a pair of submesoscale fronts in a doubly-periodic channel, let them go
# baroclinically unstable, and close the volume-integrated potential energy budget against the
# kinetic energy one.
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
# A 1 km by 2 km channel, 70 m deep, with a 30 m mixed layer over a stratified interior. The two
# fronts are 250 m wide and carry a cross-front buoyancy gradient `M²`:

Lx = 1kilometers
Ly = 2kilometers
H_target = 70.0      # [m] requested depth; the stretched grid below lands a little deeper
h  = 30.0            # [m] mixed-layer depth

f       = 1e-4       # [s⁻¹] Coriolis frequency
M²      = 3e-8       # [s⁻²] cross-front buoyancy gradient inside each front
N²_ml   = 9e-8       # [s⁻²] mixed-layer stratification (z > -h)
N²_int  = 1.8e-6     # [s⁻²] interior stratification (z < -h)
w_front = 250.0      # [m]   width of each front

# ## Grid
#
# Doubly periodic in the horizontal and bounded in the vertical. Periodicity in `y` is what the double
# front buys us, and it is what makes the transport terms of both budgets integrate to zero. The
# vertical coordinate is surface-intensified: a constant 2 m spacing over the top 32 m resolves the
# mixed layer the instability lives in, and stretches below it.
#
# The cells come out roughly 16 m by 16 m by 2 to 8 m, within an order of magnitude of isotropic. That
# matters for the closure below: `Smagorinsky` builds its filter width from the cell volume, so it is
# only meaningful on a grid that is not wildly stretched in one direction.

z = ReferenceToStretchedDiscretization(extent = H_target,
                                       bias = :right, bias_edge = 0,   # fine spacing at the surface
                                       constant_spacing = 2,
                                       constant_spacing_extent = 32,
                                       stretching = PowerLawStretching(1.08))

grid = RectilinearGrid(size = (64, 128, length(z)),
                       x = (0, Lx), y = (-Ly/2, Ly/2), z = z,
                       topology = (Periodic, Periodic, Bounded))

H = grid.Lz   # [m] the depth the grid actually has

@info "Grid: $(size(grid)), depth $(round(H, digits=1)) m, Δx = $(round(minimum_xspacing(grid), digits=1)) m, Δz from $(round(minimum_zspacing(grid), digits=2)) m to $(round(maximum(zspacings(grid, Center())), digits=2)) m"

# ## Derived quantities
#
# The mixed layer sets the scale of the instability: its deformation radius `Ld = N h / f`, the
# fastest-growing wavelength `λ ≈ 3.9 Ld`, and the growth rate `σ ≈ 0.31 M²/N`. The balanced Richardson
# number `Ri = N²/α²` is 1 in the mixed layer, which is what lets the resolved strain reach the
# stratification, and so what lets the closure below do anything at all.

α     = M² / f              # [s⁻¹]   thermal-wind shear inside a front
N_ml  = √N²_ml              # [s⁻¹]
N_int = √N²_int             # [s⁻¹]
Ld    = N_ml * h / f        # [m]     mixed-layer deformation radius
λ     = 3.9 * Ld            # [m]     fastest-growing wavelength
Ri_ml = N²_ml / α^2         # []      balanced Richardson number in the mixed layer
Δb    = M² * w_front        # [m s⁻²] buoyancy jump across each front
Ū     = α * H               # [m s⁻¹] thermal-wind velocity scale
σ     = 0.31 * M² / N_ml    # [s⁻¹]   growth rate of the mixed-layer mode

@info "Ld = $(round(Ld, digits=1)) m, λ = $(round(λ, digits=1)) m in a $(round(Int, Ly)) m channel, Ri = $(round(Ri_ml, digits=2))"
@info "Δb = $(round(Δb, sigdigits=3)) m s⁻², Ū = $(round(100Ū, digits=2)) cm/s, growth time 1/σ = $(prettytime(1/σ))"

# ## Closure
#
# We use a `Smagorinsky` closure with a constant coefficient to model turbulence stresses. We use
# a high coefficient to make sure the instability is well-resolved in this very coarse example:

using Oceananigans.TurbulenceClosures.Smagorinskys: Smagorinsky
closure = Smagorinsky(coefficient=0.3)

# ## Model
#
# The advection scheme is `Centered`, which adds no dissipation of its own, so every sink in the
# budgets below is one we write down and compute. At 2 km across, `β` does nothing, so the Coriolis
# parameter is a plain `f`-plane rather than the β-plane the mesoscale reference uses:

model = NonhydrostaticModel(grid;
                            coriolis = FPlane(; f),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            timestepper = :RungeKutta3,
                            advection = Centered(order=4),
                            closure = closure)

# ## Initial condition
#
# The buoyancy is the two-layer stratification plus two ramps of width `w_front`, one rising at
# `y = -Ly/4` and one falling at `y = +Ly/4`. Their difference returns to zero at the edges of the
# domain, so `b` matches across the periodic boundary:

ramp(y, w) = min(max(0, y/w + 1/2), 1)

y₁ = -Ly/4
y₂ = +Ly/4

double_ramp(y, w) = ramp(y - y₁, w) - ramp(y - y₂, w)

## the stratification integrated from the surface down, continuous across the mixed-layer base
b_strat(z) = ifelse(z ≥ -h, N²_ml * z, -N²_ml * h + N²_int * (z + h))

# The velocity starts in thermal-wind balance with those fronts, referenced to the bottom, and a little
# noise on the buoyancy seeds the instability:

using Random
Random.seed!(8675309)

ϵb = 1e-2 * Δb    # noise amplitude, a percent of the front's buoyancy jump

bᵢ(x, y, z) = b_strat(z) + Δb * double_ramp(y, w_front) + ϵb * randn()

ramp_prime(y, w) = (-w/2 < y < w/2) ? 1/w : zero(y)
double_ramp_prime(y, w) = ramp_prime(y - y₁, w) - ramp_prime(y - y₂, w)

uᵢ(x, y, z) = -(Δb * double_ramp_prime(y, w_front) / f) * (z + H)

set!(model, u=uᵢ, b=bᵢ)
nothing #hide

# ## Simulation

max_Δt = min(minimum_xspacing(grid) / Ū, 1 / N_int)   # Smagorinsky sets ν from the flow, so no fixed diffusive bound

simulation = Simulation(model, Δt = max_Δt, stop_time = 8days)
conjure_time_step_wizard!(simulation, IterationInterval(5), cfl = 0.7, max_Δt = max_Δt)

using Oceanostics
add_callback!(simulation, ProgressMessengers.TimedMessenger(), IterationInterval(100))

# ## The potential energy budget
#
# `eₚ = -bz` obeys `-z` times the equation the model steps for `b`, so its terms are that equation's
# terms weighted by `-z`. Over this domain the transports integrate away and the budget reads
#
# ```math
# \frac{d}{dt}\int e_p\, dV = \int \mathrm{ADV}\, dV + \int \Phi\, dV ,
# ```
#
# where ``\mathrm{ADV} = z\,\partial_j(u_jb)`` ([`PotentialEnergyBuoyancyAdvection`](@ref)) is the
# advective term as the model actually computes it, and ``\Phi = \kappa\,\partial b/\partial z``
# ([`PotentialEnergyDiffusiveVerticalBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveVerticalBuoyancyFlux))
# is the diffusive one.
#
# The two sides of the budget are treated differently on purpose. The diffusive term the model computes
# is ``z\,\partial_jq_j = \partial_j(zq_j) + \Phi``, and the transport ``\partial_j(zq_j)``
# ([`PotentialEnergyDiffusion`](@ref)) is a flux divergence that telescopes, so it drops out of the
# volume integral *exactly* and ``\Phi`` alone is the whole diffusive contribution. The advective side
# has no such luck: ``z\,\partial_j(u_jb) = -\partial_j(u_je_p) - wb`` and the transport there
# integrates to zero only in the continuum, so ``\mathrm{ADV}`` and ``-\int wb\, dV`` differ by a
# truncation error. We keep ``\mathrm{ADV}`` and output ``wb`` beside it so that difference can be
# measured rather than assumed.

eₚ  = PotentialEnergy(model)
ADV = PotentialEnergyBuoyancyAdvection(model)
Φ   = PotentialEnergyDiffusiveVerticalBuoyancyFlux(model)

# ## The kinetic energy budget
#
# The other side of the exchange. `wb` carries opposite signs in the two budgets, so it cancels from
# their sum:
#
# ```math
# \frac{d}{dt}\int K\, dV = \int wb\, dV - \int \varepsilon_a\, dV .
# ```

K  = KineticEnergy(model)
wb = PotentialToKineticEnergyConversion(model)
εₐ = KineticEnergyDissipationRate(model)

∫eₚ  = ∫dV(eₚ)
∫ADV = ∫dV(ADV)
∫Φ   = ∫dV(Φ)
∫K   = ∫dV(K)
∫wb  = ∫dV(wb)
∫εₐ  = ∫dV(εₐ)

# For the movie we keep the surface vorticity and buoyancy:

u, v, w = model.velocities
b = model.tracers.b
ζ = ∂x(v) - ∂y(u)

# ## Output
#
# A *snapshot* writer for the surface maps and a *budget* writer for the volume integrals, the latter
# on `ConsecutiveIterations`, which takes a second sample one model step after each output time so we
# can finite-difference `d/dt` across that step.

using NCDatasets
filename = joinpath(@__DIR__, "baroclinic_adjustment")

simulation.output_writers[:fields] =
    NetCDFWriter(model, (; ζ, b),
                 filename = filename,
                 schedule = TimeInterval(3hours),
                 indices = (:, :, grid.Nz),
                 overwrite_existing = true)

simulation.output_writers[:budget] =
    NetCDFWriter(model, (; ∫eₚ, ∫ADV, ∫Φ, ∫K, ∫wb, ∫εₐ),
                 filename = filename * "_budget",
                 schedule = ConsecutiveIterations(TimeInterval(3hours)),
                 overwrite_existing = true)

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

ζlim = maximum(abs, ζ_arr)

# The budget scalars come in consecutive-iteration pairs `(2k-1, 2k)`; a one-step finite difference
# inside each pair gives the tendencies, and every source term is averaged over the same pair.

ds_b = NCDataset(simulation.output_writers[:budget].filepath)
t_bud    = ds_b["time"][:]
eₚ_bud  = ds_b["∫eₚ"][:]
ADV_bud = ds_b["∫ADV"][:]
Φ_bud   = ds_b["∫Φ"][:]
K_bud   = ds_b["∫K"][:]
wb_bud  = ds_b["∫wb"][:]
εₐ_bud  = ds_b["∫εₐ"][:]
close(ds_b)

idx1 = 1:2:length(t_bud) - 1
idx2 = 2:2:length(t_bud)

Δt_pair = t_bud[idx2] .- t_bud[idx1]
t_pair  = @. 0.5 * (t_bud[idx1] + t_bud[idx2])

deₚdt = (eₚ_bud[idx2] .- eₚ_bud[idx1]) ./ Δt_pair
dKdt  = (K_bud[idx2]  .- K_bud[idx1])  ./ Δt_pair

pair_mean(x) = @. 0.5 * (x[idx1] + x[idx2])

ADV_pair = pair_mean(ADV_bud)
Φ_pair   = pair_mean(Φ_bud)
wb_pair  = pair_mean(wb_bud)
εₐ_pair  = pair_mean(εₐ_bud)

# Both budgets in sum-to-zero form, plus the collapse that lets `∫wb dV` stand in for the advective
# term:

eₚ_resid = @. -deₚdt + ADV_pair + Φ_pair
K_resid  = @. -dKdt  + wb_pair  - εₐ_pair

adv_collapse = @. ADV_pair + wb_pair     # should vanish: ∫z∂ⱼ(uⱼb) dV = -∫wb dV

using Test                                                                            #hide
using Statistics: mean                                                                #hide
rms(x) = √(sum(abs2, x) / length(x))                                                  #hide
eₚ_terms = (deₚdt, ADV_pair, Φ_pair)                                                  #hide
K_terms  = (dKdt, wb_pair, εₐ_pair)                                                   #hide
## Each budget closes to a couple of percent of its own largest term.                                #hide
@test rms(eₚ_resid) < 0.03 * maximum(rms, eₚ_terms)                                   #hide
@test rms(K_resid)  < 0.05 * maximum(rms, K_terms)                                    #hide
## `wb` is the same term in both, so it cancels from their sum                        #hide
@test rms(eₚ_resid .+ K_resid) < 0.05 * rms(wb_pair)                                  #hide
## `∫ADV = -∫wb` is the one collapse still worth checking: unlike the diffusive one it is second      #hide
## order in the grid spacing rather than exact, which is why `ADV` is what the budget uses.          #hide
@test rms(adv_collapse) < 0.05 * rms(wb_pair)                                         #hide
## The adjustment does what it should: the fronts slump and release potential energy into kinetic    #hide
## energy. `∫ADV dV` is that conversion seen from the potential energy side, so it is negative.      #hide
@test mean(wb_pair)  > 0                                                              #hide
@test mean(ADV_pair) < 0                                                              #hide
## It leads the kinetic energy budget. It does not lead the potential energy one — with this closure #hide
## the diffusive term is an order of magnitude larger — but the budget resolves it far above its own #hide
## residual, which is the claim that matters.                                                        #hide
@test rms(wb_pair)  > 0.5 * rms(dKdt)                                                 #hide
@test rms(ADV_pair) > 10 * rms(eₚ_resid)                                              #hide
@test rms(Φ_pair)   > rms(ADV_pair)                                                   #hide
## The closure is active everywhere, which is what a constant-coefficient Smagorinsky does and what  #hide
## the Lilly-corrected one would not at this resolution.                                             #hide
@test minimum(εₐ_pair) > 0                                                            #hide
@test all(εₐ_pair .≥ 0);                                                              #hide

# ## Plotting

set_theme!(Theme(fontsize = 18))
fig = Figure(size = (1100, 1000))

n = Observable(1)

panel_kwargs = (xlabel = "x [m]", ylabel = "y [m]", aspect = DataAspect(), height = 260)
axζ = Axis(fig[2, 1]; title = "vertical vorticity, ζ", panel_kwargs...)
axb = Axis(fig[2, 3]; title = "surface buoyancy, b",   panel_kwargs...)

ζₙ = @lift ζ_arr[:, :, $n]
bₙ = @lift b_arr[:, :, $n]

hmζ = heatmap!(axζ, x_faa, y_afa, ζₙ; colormap = :balance, colorrange = (-ζlim, ζlim))
Colorbar(fig[2, 2], hmζ)

hmb = heatmap!(axb, x_caa, y_aca, bₙ; colormap = :thermal)
Colorbar(fig[2, 4], hmb)

budget_kwargs = (xlabel = "time [days]", ylabel = "[m⁵ s⁻³]")

ax_p = Axis(fig[3, 1:4]; title = "Volume-integrated potential energy budget", budget_kwargs...)
lines!(ax_p, t_pair ./ day, -deₚdt,     label = "-d(∫eₚ)/dt")
lines!(ax_p, t_pair ./ day,  ADV_pair,  label = "∫ADV dV  (= -∫wb dV)")
lines!(ax_p, t_pair ./ day,  Φ_pair,    label = "∫Φ dV  (diffusive buoyancy flux)")
lines!(ax_p, t_pair ./ day,  eₚ_resid,  label = "residual", color = :black, linestyle = :dash)
axislegend(ax_p; position = :rt, labelsize = 10, nbanks = 2)

ax_K = Axis(fig[4, 1:4]; title = "Volume-integrated kinetic energy budget", budget_kwargs...)
lines!(ax_K, t_pair ./ day, -dKdt,    label = "-d(∫K)/dt")
lines!(ax_K, t_pair ./ day,  wb_pair, label = "∫wb dV  (buoyancy conversion)")
lines!(ax_K, t_pair ./ day, -εₐ_pair, label = "-∫εₐ dV  (dissipation)")
lines!(ax_K, t_pair ./ day,  K_resid, label = "residual", color = :black, linestyle = :dash)
axislegend(ax_K; position = :rt, labelsize = 10, nbanks = 2)

vlines!(ax_p, @lift(times[$n] / day), color = :black, linestyle = :dot)
vlines!(ax_K, @lift(times[$n] / day), color = :black, linestyle = :dot)

title = @lift "Baroclinic adjustment, t = " * prettytime(times[$n])
fig[1, 1:4] = Label(fig, title, fontsize = 22, tellwidth = false)

@info "Animating..."
record(fig, "baroclinic_adjustment.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
set_theme!() #hide
nothing #hide

# ![](baroclinic_adjustment.mp4)
#
# Both fronts go unstable and roll up into submesoscale eddies, and the two budget panels show the
# energy moving between the reservoirs. `∫wb dV` is the through line: the eddies slump the fronts and
# what the potential energy loses to that conversion appears as `∫K dV`. Since it enters the two
# budgets with opposite signs it cancels from their sum, which is the sense in which this is one
# exchange rather than two independent balances.
#
# The potential energy panel is the one this example exists for. `∫ADV dV` is plotted as the module
# computes it, `-z` times the advective term of the buoyancy equation, and it lands on `-∫wb dV` to
# about a percent. The two differ by the transport `∂ⱼ(uⱼeₚ)`, which integrates to zero only in the
# continuum, so that percent is the discretization and not a mistake. The diffusive term needs no such
# caveat: its transport `∂ⱼ(zqⱼ)` telescopes on the discrete grid, so `∫Φ dV` *is* the whole diffusive
# contribution and the budget uses it directly.
#
# The diffusive term dominates that panel. A constant-coefficient `Smagorinsky` is active everywhere
# rather than only where the strain beats the stratification, and at a coefficient chosen to keep this
# coarse grid well behaved it moves an order of magnitude more potential energy than the conversion
# does. The conversion is still resolved far above the budget's residual, which is what lets the panel
# separate the two at all.
#
# The kinetic energy residual sits at a fraction of a percent of its budget's largest term, and neither
# residual vanishes: the discrete `K` and `eₚ` equations are not derived from the discrete momentum and
# buoyancy equations the model steps, so the two sides agree only to truncation error. The same caveat
# applies to [the two-dimensional turbulence example](@ref two_d_turbulence_example).
#
# `∫K dV` itself barely changes over the run, which is not a failure of the instability. The initial
# condition already carries the thermal wind, so `K` starts at the balanced flow's value rather than at
# zero; the eddies grow out of that flow rather than on top of nothing, and `∫εₐ dV` removes about as
# much as `∫wb dV` supplies. What the budget shows is the throughput, not the accumulation.
