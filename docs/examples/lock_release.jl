# # [Lock release and the sorted reference state](@id lock_release_example)
#
# In this example we run a two-dimensional lock release simulation and use it to watch the available potential
# energy and kinetic energy evolve. Along the way we build the reference profile
# with each of the four methods Oceanostics provides and close
# the volume-integrated available potential energy and kinetic energy budgets against the two dissipation
# rates and the buoyancy production that connects them.
#
# Before starting, make sure you have the required packages installed for this example, which can be
# done with
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Oceanostics, CairoMakie"
# ```

# ## Model and simulation setup

using Oceananigans

# We work with nondimensional quantities. The buoyancy jump across the lock `Δb` and the channel depth
# `H` set the buoyancy velocity `U = √(Δb H) / 2`, which is the classic lock-release front speed and
# the only velocity scale in the problem. The channel is four times as long as it is deep:

Δb = 1      # buoyancy jump across the lock
H  = 1      # channel depth
Lx = 4H     # channel length

U = √(Δb * H) / 2   # buoyancy velocity

# The domain is walled at both ends and at top and bottom, so the fronts eventually reflect and the
# channel fills with a mixed intermediate layer. That is what we want here: it drives the reference
# profile all the way from a step to something smooth. The grid is isotropic at `Δ = H/Nz`:

Nz = 128
grid = RectilinearGrid(size = (4Nz, Nz), x = (-Lx/2, Lx/2), z = (0, H), topology = (Bounded, Flat, Bounded))

# The closure and the advection scheme are chosen to facilitate closing the budgets at the end: we
# choose a centered scheme, which adds no dissipation of its own and constant diffusivity with
# `ν` and `κ` large enough that the grid resolves whatever the flow makes of the front.

ν = κ = 5e-4
model = NonhydrostaticModel(grid; timestepper = :RungeKutta3,
                            advection = Centered(order=4), # Non-dissipative scheme
                            closure = ScalarDiffusivity(; ν, κ),
                            buoyancy = BuoyancyTracer(), tracers = :b);

# The lock itself: buoyant fluid on the right, dense fluid on the left, separated by a thin interface.
# Smoothing the step over `δ` keeps the initial condition off the grid scale:

δ = 4 * minimum_xspacing(grid)          # interface thickness, four cells
lock_release(x, z) = (Δb / 2) * tanh(x / δ)
set!(model, b = lock_release)

# We create a simulation with conservative choices for the initial time step and the CFL number.
# Since we anticipate the flow to be fairly viscous, we also set a target diffusive CFL:

simulation = Simulation(model, Δt = 0.1 * minimum_xspacing(grid) / U, stop_time = 20)
conjure_time_step_wizard!(simulation, IterationInterval(2), cfl = 0.7, diffusive_cfl = 0.2)

using Oceanostics
progress = ProgressMessengers.BasicMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(500));

# ## Diagnostics
#
# We build one `z✶` per method and write all of them for later comparison. They are ordinary `Field`s
# that re-sort themselves whenever they are computed, so they can be passed to an output writer.
# Given that the reference height calculation is nonlocal, it is generally a computationally-heavy operation
# and its cost grows faster than the number of cells: see [Computational cost per method](@ref) for how different
# methods compare and how they scale.

b = model.tracers.b

z✶_ranked    = reference_height(model, method = ThreeDimensionalSort())
z✶_heaviside = reference_height(model, method = HeavisideIntegral())
z✶_lookup    = reference_height(model, method = ProfileLookup())
z✶_column    = reference_height(model, method = VerticalSort())
b✶_column    = reference_buoyancy(z✶_column)

# Both background and available potential energies are built from the same reference height, so we share one rather than letting each
# diagnostic sort the domain for itself. Note that `APE` here is the *local* available potential energy
# (as defined by Holliday & McIntyre (1981)) and it should always be non-negative.
#
# We build one APE calculation per method. Three of the four answer on the model grid and give a map over the
# domain, which is what the animation below compares. [`VerticalSort`](@ref) is given on the sorted
# column, so its APE is ordered by rank rather than by position and is not a map of the flow; it is
# written anyway because the volume integral comes out the same whichever method builds the reference
# state. [`ProfileLookup`](@ref) is that same column read back onto the model grid, matching each cell
# to the profile through its buoyancy rather than through where it came from.

APE_ranked    = AvailablePotentialEnergy(model, z✶_ranked)
APE_heaviside = AvailablePotentialEnergy(model, z✶_heaviside)
APE_lookup    = AvailablePotentialEnergy(model, z✶_lookup)
APE_column    = AvailablePotentialEnergy(model, z✶_column)
KE            = KineticEnergy(model)

∫BPE = Integral(BackgroundPotentialEnergy(model, z✶_ranked))
∫KE  = Integral(KE)

∫APE           = Integral(APE_ranked)
∫APE_heaviside = Integral(APE_heaviside)
∫APE_lookup    = Integral(APE_lookup)
∫APE_column    = Integral(APE_column)

# ## Budget terms
#
# In a closed box the volume-integrated kinetic and available potential energies exchange through a
# single term and each drains through a dissipation of its own,
#
# ```math
# \frac{d}{dt}\int e_k\, dV = +\int w b_r\, dV - \int \varepsilon_k\, dV, \qquad
# \frac{d}{dt}\int e_a\, dV = -\int w b_r\, dV - \int \varepsilon_a\, dV .
# ```
#
# Advection and the pressure gradient integrate to zero for an incompressible flow in a closed box, and
# the walls are free-slip and insulating, so the viscous and diffusive fluxes leave nothing at the
# boundary either. What survives is the exchange between the two reservoirs and the two sinks.
#
# Note that the exchange term is written ``w b_r`` ([`AvailablePotentialToKineticEnergyConversion`](@ref))
# in both budgets, not ``wb`` ([`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion)),
# where ``b_r = b - b^\star(z)`` is the buoyancy anomaly relative to the reference profile at the
# parcel's own height, representing an exchange between kinetic energy and _available potential_
# energy rather than _potential_ energy. This is possible in the KE equation since ``b^\star`` is a
# function of height alone, so the hydrostatic pressure that goes with it, ``p^\star(z)`` with
# ``\partial p^\star/\partial z = b^\star``, has no horizontal gradient and can be absorbed into the
# pressure term. That said, both terms are equal in an integrated sense (as we'll show below).
#
# The dissipation term ``\varepsilon_a`` is the contraction of the closure's diffusive buoyancy
# flux with the gradient of the displacement potential
# [``\Upsilon = z^\star - z``](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDisplacementPotential):
#
# ```math
# \varepsilon_a = -q_i\, \partial_i \Upsilon
# ```
# where ``q_i`` is whatever diffusive buoyancy flux the closure supplies.
#
# We hand ``\varepsilon_a`` the [`HeavisideIntegral`](@ref) reference height already built above rather
# than letting it sort the domain again, and that method rather than another because
# ``\varepsilon_a`` differentiates the map ``\Upsilon``: Eq. (11) of Winters et al. is the one that makes
# ``z^\star`` a function of buoyancy alone, so tied cells do not spread ``z^\star`` over the depth they
# fill and show up in ``\nabla \Upsilon`` as grid-scale noise.

εₐ  = AvailablePotentialEnergyDissipationRate(model, z✶_heaviside)
εₖ  = KineticEnergyDissipationRate(model)
wbᵣ = AvailablePotentialToKineticEnergyConversion(model, z✶_heaviside)

# Let's also calculate the exchange between kinetic and potential energy ``wb`` for comparison
# along with integrals for all terms

wb = PotentialToKineticEnergyConversion(model)

∫wbᵣ = Integral(wbᵣ)
∫wb  = Integral(wb)
∫εₖ  = Integral(εₖ)
∫εₐ  = Integral(εₐ)

using NCDatasets
filename = "lock_release"

# Two writers. The *fields* writer carries the maps, and a single `NetCDFWriter` copes with the two
# grids they live on: the model-grid fields are written against `x` and `z`, and the column's against
# its own `N`-cell vertical axis.

outputs = (; b, KE, APE_ranked, APE_heaviside, APE_lookup, APE_column,
             z✶_ranked, z✶_heaviside, z✶_lookup, z✶_column, b✶_column, wb, wbᵣ)

simulation.output_writers[:fields] = NetCDFWriter(model, outputs,
                                                  filename = joinpath(@__DIR__, filename),
                                                  schedule = TimeInterval(0.5),
                                                  overwrite_existing = true)

# The *budget* writer carries only the volume integrals, which are cheap next to the maps, on
# `ConsecutiveIterations(TimeInterval(0.5))`. That schedules a second sample one model step after each
# scheduled time, which lets us finite-difference ``d/dt`` across that step instead of accumulating it.
# The `∫APE_heaviside` written here is the tendency term that pairs with ``\varepsilon_a``, since the
# two come off the same sort.

integrals = (; ∫BPE, ∫KE, ∫APE, ∫APE_heaviside, ∫APE_lookup, ∫APE_column, ∫wb, ∫wbᵣ, ∫εₖ, ∫εₐ)

simulation.output_writers[:budget] = NetCDFWriter(model, integrals,
                                                  filename = joinpath(@__DIR__, filename * "_budget"),
                                                  schedule = ConsecutiveIterations(TimeInterval(0.5)),
                                                  overwrite_existing = true)

# ## Run the simulation

run!(simulation)

# ## Reference profile
#
# The reference state is what you get by rearranging every parcel adiabatically into the state of
# minimum potential energy: rank the cells by buoyancy and stack them from the bottom of the domain up.
# The height a parcel lands at is its reference height ``z^\star``, and plotting the buoyancy that goes
# with it gives the reference profile ``b^\star(z^\star)`` — the stratification the flow would have if
# all of its available potential energy were released.
#
# Everything needed is already in the file. For the column the profile is written as it stands. For the
# two model-grid methods, `z✶` and `b` are both maps over the cells, so pairing them and ordering by
# `z✶` recovers the same profile; that reordering is all the "post-processing" amounts to.

using Oceananigans.Fields: interior

filepath = simulation.output_writers[:fields].filepath
b_t = FieldTimeSeries(filepath, "b")     # for the movie below

ds = NCDataset(filepath)
times = ds["time"][:]
B  = ds["b"][:, :, :]                    # (x, z, time); y is Flat, so it is dropped
Z3 = ds["z✶_ranked"][:, :, :]
ZH = ds["z✶_heaviside"][:, :, :]
Z1 = ds["z✶_column"][:, :, :]            # (1, N, time): the column keeps the model's Flat y, dropped here
B1 = ds["b✶_column"][:, :, :]
close(ds)

## pair a reference-height map with the buoyancy map and order by z✶
mapped_profile(Z, n) = (h = vec(Float64.(Z[:, :, n])); p = sortperm(h); (vec(Float64.(B[:, :, n]))[p], h[p]))

## the column is already ordered, so it is read straight off
column_profile(n) = (vec(Float64.(B1[:, :, n])), vec(Float64.(Z1[:, :, n])))

snapshot_times = [0, 5, 10, 20]
snapshots = [argmin(abs.(times .- t)) for t in snapshot_times]

methods = ("ThreeDimensionalSort" => n -> mapped_profile(Z3, n),
           "HeavisideIntegral"    => n -> mapped_profile(ZH, n),
           "VerticalSort"         => column_profile)

## `profiles[name][k]` is the `(b✶, z✶)` pair for method `name` at the `k`-th snapshot
profiles = Dict(name => [build(n) for n in snapshots] for (name, build) in methods);

# All three describe the same reference state, so wherever the buoyancy field is continuous their
# profiles coincide. They part ways only where cells are *tied* at exactly the same buoyancy, and a
# lock is the extreme case: the initial condition saturates to `±Δb/2` away from the interface, leaving
# just a few dozen distinct buoyancies across tens of thousands of cells.

n_distinct(n) = length(unique(vec(Float64.(B[:, :, n]))))
@info "distinct buoyancies: $(n_distinct(snapshots[1])) at t = 0, " *
      "$(n_distinct(snapshots[end])) at t = $(times[snapshots[end]]), of $(prod(size(grid))) cells"

using Test                                                                                #hide
## `b✶_column` is written *during* the run, so each snapshot has to be a                  #hide
## permutation of `b` at the same time. This is what catches a reference                  #hide
## state that lags an output behind the flow.                                             #hide
for n in snapshots                                                                        #hide
    @test sort(vec(Float64.(B1[:, :, n]))) ≈ sort(vec(Float64.(B[:, :, n]))) atol=1e-5    #hide
end                                                                                       #hide
z✶_3d_0,  z✶_hv_0 = profiles["ThreeDimensionalSort"][1][2], profiles["HeavisideIntegral"][1][2]   #hide
b✶_3d, z✶_3d = profiles["ThreeDimensionalSort"][end]                                      #hide
b✶_hv, z✶_hv = profiles["HeavisideIntegral"][end]                                         #hide
b✶_1d, z✶_1d = profiles["VerticalSort"][end]                                        #hide
## the two model-grid methods differ by a sixth of the depth while the lock is intact     #hide
@test maximum(abs, z✶_hv_0 .- z✶_3d_0) > 0.15H                                            #hide
## and agree to the grid scale once mixing has made the field continuous                  #hide
@test maximum(abs, z✶_hv .- z✶_3d) < H / Nz                                               #hide
## the column carries exactly what the ranked sort assigns, by construction               #hide
@test maximum(abs, z✶_3d .- z✶_1d) < 1e-12                                                #hide
@test maximum(abs, b✶_3d .- b✶_1d) < 1e-12                                                #hide
## and the reference profile is, by construction, monotonic                               #hide
@test issorted(b✶_1d)                                                                     #hide
## The three methods describe one reference state, so although their `z✶` maps differ wherever    #hide
## cells are tied, every buoyancy-weighted integral of `z✶` has to agree — that integral is `BPE`, #hide
## and its independence from the method is the precise sense in which the three reference heights  #hide
## are the same. Unlike the pointwise comparisons above it holds at every time, ties or not, so it #hide
## is checked at all four snapshots rather than only once the field has gone continuous.           #hide
for n in snapshots                                                                        #hide
    bn, b1n = vec(Float64.(B[:, :, n])), vec(Float64.(B1[:, :, n]))                       #hide
    E_ranked    = sum(bn  .* vec(Float64.(Z3[:, :, n])))                                  #hide
    E_heaviside = sum(bn  .* vec(Float64.(ZH[:, :, n])))                                  #hide
    E_column    = sum(b1n .* vec(Float64.(Z1[:, :, n])))                                  #hide
    @test E_heaviside ≈ E_ranked rtol=1e-8                                                #hide
    @test E_column    ≈ E_ranked rtol=1e-8                                                #hide
    ## the two that give every cell its own slot assign exactly the same set of heights   #hide
    @test sort(vec(Float64.(Z3[:, :, n]))) ≈ sort(vec(Float64.(Z1[:, :, n]))) atol=1e-6   #hide
end                                                                                       #hide
## `HeavisideIntegral` is the one that can differ pointwise, and only while ties survive: it is    #hide
## a sixth of the depth out at `t = 0` and converges to the others as mixing removes the ties.     #hide
Δ_hv(n) = maximum(abs, vec(Float64.(ZH[:, :, n])) .- vec(Float64.(Z3[:, :, n])))          #hide
@test Δ_hv(snapshots[1])   > 0.15H                                                        #hide
@test Δ_hv(snapshots[end]) < H / Nz;                                                      #hide

# The profile gets a figure of its own: one panel per method, each showing `b★(z✶)` at the four times
# above. It starts as a step, two blocks of uniform buoyancy stacked one on the other, and mixing
# erodes it into a smooth stratification.

using CairoMakie

set_theme!(Theme(fontsize = 20))
fig = Figure();

colors = cgrad(:viridis, length(snapshots); categorical = true)

for (m, (name, _)) in enumerate(methods)
    row, col = fldmod1(m, 2)   # fill a 2×2 grid row by row
    local ax = Axis(fig[row, col]; xlabel = "b✶", title = name, width = 200, height = 280,
                    ylabel = col == 1 ? "z✶" : "", yticklabelsvisible = col == 1, titlesize = 15)
    ylims!(ax, 0, H)
    for (s, n) in enumerate(snapshots)
        b✶, z✶ = profiles[name][s]
        lines!(ax, b✶, z✶; color = colors[s], linewidth = 2, label = "t = $(round(times[n], digits=1))")
    end
    m == 1 && axislegend(ax; position = :lt, labelsize = 11)
end

# The fourth panel puts the three side by side at `t = 0`, where they disagree the most.
# [`HeavisideIntegral`](@ref) is drawn as markers rather than a line because its `z✶` takes only as many
# distinct values as there are distinct buoyancies, which is a few dozen here against 65536 cells.

row0, col0 = fldmod1(length(methods) + 1, 2)
ax0 = Axis(fig[row0, col0]; xlabel = "b✶", title = "t = 0, all three", width = 200, height = 280,
           ylabel = col0 == 1 ? "z✶" : "", yticklabelsvisible = col0 == 1, titlesize = 15)
ylims!(ax0, 0, H)

b✶_r, z✶_r = profiles["ThreeDimensionalSort"][1]
b✶_c, z✶_c = profiles["VerticalSort"][1]
b✶_h, z✶_h = profiles["HeavisideIntegral"][1]

lines!(ax0, b✶_r, z✶_r; linewidth = 5, color = (:steelblue, 0.9), label = "ThreeDimensionalSort")
lines!(ax0, b✶_c, z✶_c; linewidth = 2, linestyle = :dash, color = :black, label = "VerticalSort")
scatter!(ax0, b✶_h, z✶_h; markersize = 9, color = :crimson, label = "HeavisideIntegral")
axislegend(ax0; position = :lt, labelsize = 9)

resize_to_layout!(fig)
save("lock_release_profiles.png", fig)
set_theme!() #hide
nothing #hide

# The three method panels are identical except while the lock is still intact, and that difference is
# informative rather than an error. At `t = 0` almost every cell is tied with thousands of others at
# one of two buoyancies, and the methods place tied cells differently.
# [`ThreeDimensionalSort`](@ref) and [`VerticalSort`](@ref) give each cell its own slot in the
# stack, so they draw the true step spanning the full depth. [`HeavisideIntegral`](@ref) instead
# collapses each buoyancy class onto the mid-height of the layer it fills, which is what makes `z✶` a
# function of buoyancy alone and a clean field to map, but leaves it unable to represent a step as a
# profile: its `z✶` only ever reaches the mid-heights of the two blocks, about a sixth of the depth
# in from each boundary. Once mixing has made the buoyancy field continuous the ties vanish and all
# three agree to within a grid cell.

# ![](lock_release_profiles.png)

# ## Flow animation and local energies
#
# The movie sets the flow beside the energy it carries: buoyancy, kinetic energy, and the local `eₐ`
# built from each of the three methods that answer on the model grid.

KE_t   = FieldTimeSeries(filepath, "KE")
APE3_t = FieldTimeSeries(filepath, "APE_ranked")
APEH_t = FieldTimeSeries(filepath, "APE_heaviside")
APEL_t = FieldTimeSeries(filepath, "APE_lookup")

## the local form is non-negative everywhere, whichever method builds the reference state          #hide
for A in (APE3_t, APEH_t, APEL_t)                                                                  #hide
    @test minimum(minimum(interior(A[n])) for n in 1:length(times)) ≥ -1e-6 * maximum(interior(A[end]))  #hide
end                                                                                                #hide
## the three model-grid methods agree cell by cell, not merely in the integral                     #hide
for n in 1:length(times)                                                                           #hide
    @test interior(APEH_t[n]) ≈ interior(APE3_t[n]) atol=1e-12                                     #hide
    @test interior(APEL_t[n]) ≈ interior(APE3_t[n]) atol=1e-12                                     #hide
end                                                                                                #hide

fig3 = Figure(size = (900, 1010))

n = Observable(1)

## `DataAspect` draws the channel at its true proportions, so a `4H` by `H` domain comes out four times
## as wide as it is deep. The width follows from the height, rather than both being set independently.
panel_kwargs = (ylabel = "z", height = 190, aspect = DataAspect())

ax_b  = Axis(fig3[2, 1];  title = "Buoyancy b",                      panel_kwargs...)
ax_KE = Axis(fig3[4, 1];  title = "Kinetic energy",                  panel_kwargs...)
ax_E3 = Axis(fig3[6, 1];  title = "eₐ,  ThreeDimensionalSort",       panel_kwargs...)
ax_EH = Axis(fig3[8, 1];  title = "eₐ,  HeavisideIntegral",          panel_kwargs...)
ax_EL = Axis(fig3[10, 1]; title = "eₐ,  ProfileLookup", xlabel = "x", panel_kwargs...)

bₙ  = @lift b_t[$n]
KEₙ = @lift KE_t[$n]
E3ₙ = @lift APE3_t[$n]
EHₙ = @lift APEH_t[$n]
ELₙ = @lift APEL_t[$n]

## `eₐ` and the kinetic energy are both sign-definite, so they get one-sided ranges set from their own
## peak over the run; the buoyancy keeps the symmetric range used above.
KE_lim = maximum(maximum(interior(KE_t[k]))   for k in 1:length(times))
Ea_lim = maximum(maximum(interior(APE3_t[k])) for k in 1:length(times))

hm_b  = heatmap!(ax_b,  bₙ;  colormap = :balance, colorrange = (-Δb/2, Δb/2))
Colorbar(fig3[3, 1], hm_b;  vertical = false, height = 8)

energy_options = (; colormap = :magma, colorrange = (0, 0.5Ea_lim))
hm_KE = heatmap!(ax_KE, KEₙ; energy_options...)
Colorbar(fig3[5, 1], hm_KE; vertical = false, height = 8)

hm_E3 = heatmap!(ax_E3, E3ₙ; energy_options...)
Colorbar(fig3[7, 1], hm_E3; vertical = false, height = 8)

hm_EH = heatmap!(ax_EH, EHₙ; energy_options...)
Colorbar(fig3[9, 1], hm_EH; vertical = false, height = 8)

hm_EL = heatmap!(ax_EL, ELₙ; energy_options...)
Colorbar(fig3[11, 1], hm_EL; vertical = false, height = 8)

title = @lift "Lock release,  t = " * string(round(times[$n], digits = 1))
Label(fig3[1, 1], title, fontsize = 22, tellwidth = false)

resize_to_layout!(fig3)

@info "Animating..."
record(fig3, "lock_release.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
nothing #hide

# ![](lock_release.mp4)
#
# `eₐ` drains from the lock as the fronts accelerate and refills wherever the seiche lifts dense fluid
# back above its reference height. Being the local form, it is non-negative everywhere. The three `eₐ`
# panels are identical: the methods differ only in where inside a tied run they place `z✶`, and `eₐ`
# cannot see that choice.

# ## Energetics
#
# The same three energies, now volume integrated, read off the budget writer. Its samples come in
# consecutive-iteration pairs `(t₀, t₀ + Δt_model, t₀ + 0.5, t₀ + 0.5 + Δt_model, …)`, so pair `k` sits
# at indices `(2k-1, 2k)`. The first of each pair falls on the `TimeInterval` grid and is what the
# energy curves use; the pair as a whole is what gives `d/dt` further down.

ds = NCDataset(simulation.output_writers[:budget].filepath)
t_bud   = ds["time"][:]
BPE_bud = ds["∫BPE"][:]
APE_bud = ds["∫APE_heaviside"][:]
KE_bud  = ds["∫KE"][:]
wb_bud  = ds["∫wb"][:]
wbᵣ_bud = ds["∫wbᵣ"][:]
εₖ_bud  = ds["∫εₖ"][:]
εₐ_bud  = ds["∫εₐ"][:]
## all four methods integrate to the same eₐ, however they place cells of equal buoyancy  #hide
@test ds["∫APE"][:]        ≈ APE_bud rtol=1e-8                                             #hide
@test ds["∫APE_lookup"][:] ≈ APE_bud rtol=1e-8                                             #hide
@test ds["∫APE_column"][:] ≈ APE_bud rtol=1e-8                                             #hide
close(ds)

idx1 = 1:2:length(t_bud) - 1   # primary snapshots
idx2 = 2:2:length(t_bud)       # consecutive-iteration snapshots

t_e     = t_bud[idx1]
KE_int  = KE_bud[idx1]
APE_int = APE_bud[idx1]
BPE_int = BPE_bud[idx1]

total_int = KE_int .+ APE_int .+ BPE_int

@test APE_int[1] > 0                                        # a lock is pure available PE     #hide
@test minimum(APE_int) < 0.05 * APE_int[1]                  # the collapse nearly empties it  #hide
@test all(APE_int .≥ -1e-8)                                 # and it is never negative        #hide
@test KE_int[1] < 1e-8 * APE_int[1]                         # the lock starts at rest         #hide
@test BPE_int[end] > BPE_int[1]                             # mixing raised the background    #hide
@test minimum(diff(BPE_int)) > -1e-6 * maximum(abs, BPE_int) # and only ever raised it        #hide
@test total_int[end] < total_int[1]                          # dissipation outweighs diffusion #hide
## Diffusion working against gravity can feed the total back, so it need not fall monotonically. But  #hide
## an endpoint test alone would pass a run that spiked mid-simulation and decayed below its start, so #hide
## bound the rise: measured, the worst interval falls, so any rise at all is already anomalous.       #hide
@test maximum(diff(total_int)) < 0.02 * abs(total_int[1] - total_int[end]);                           #hide

set_theme!(Theme(fontsize = 20)) #hide
fig2 = Figure(size = (780, 350))
ax = Axis(fig2[1, 1]; xlabel = "Time", ylabel = "Energy", title = "Lock-release energetics")
lines!(ax, t_e, KE_int,  label = "∫KE dV")
lines!(ax, t_e, APE_int, label = "∫APE dV")
lines!(ax, t_e, BPE_int, label = "∫BPE dV")
lines!(ax, t_e, total_int; label = "total", color = :black, linestyle = :dash)
axislegend(ax; position = :rc, labelsize = 12)

save("lock_release_energetics.png", fig2)
nothing #hide

# ![](lock_release_energetics.png)
#
# `BPE` never turns back, since mixing across density surfaces cannot be undone. Everything the flow
# can still do sits in `APE`, which trades with `KE` as the box seiches, each cycle weaker than the
# last. The dashed total is `∫KE + ∫eₚ`, and it has no reason to fall monotonically: viscosity drains it
# at `∫εₖ` while the diffusive flux working against gravity feeds it back at `∫Φ dV = -∫q₃ dV`, and a run quiet enough
# for the second to win would see the total edge back up. At `Re = 1000` the first stays the larger of
# the two throughout, and the total ends down by about a fifth of the available energy the lock started
# with.

# ## Closing the budgets
#
# Now the two budgets written at the top. `d/dt` comes from a one-step finite difference inside each
# consecutive-iteration pair, and the source terms are averaged over the same pair so that every term
# is evaluated at the same instant.

Δt_pair = t_bud[idx2] .- t_bud[idx1]
t_pair  = @. 0.5 * (t_bud[idx1] + t_bud[idx2])

dKEdt  = (KE_bud[idx2]  .- KE_bud[idx1])  ./ Δt_pair
dAPEdt = (APE_bud[idx2] .- APE_bud[idx1]) ./ Δt_pair

pair_mean(x) = @. 0.5 * (x[idx1] + x[idx2])

wb_pair  = pair_mean(wb_bud);
wbᵣ_pair = pair_mean(wbᵣ_bud);
εₖ_pair  = pair_mean(εₖ_bud);
εₐ_pair  = pair_mean(εₐ_bud);

# Both budgets are written in sum-to-zero form: each curve is plotted with the sign it carries here, so
# the panels below add up to the residual.

KE_resid  = @. -dKEdt  + wbᵣ_pair - εₖ_pair
APE_resid = @. -dAPEdt - wbᵣ_pair - εₐ_pair

rms(x) = √(sum(abs2, x) / length(x))                                       #hide
@test rms(KE_resid)  < 0.01 * rms(dKEdt)                                   #hide
@test rms(KE_resid)  < 0.02 * rms(εₖ_pair)                                  #hide
@test rms(APE_resid) < 0.01 * rms(dAPEdt)                                  #hide
## the sharp one: the APE residual is a small fraction of εₐ itself, so the budget resolves     #hide
## the new term rather than closing to within its size                                           #hide
@test rms(APE_resid) < 0.05 * rms(εₐ_pair)                                #hide
## the two conversions have the same volume integral, so they cancel from the sum of the budgets #hide
@test rms(wb_pair .- wbᵣ_pair) < 1e-8 * rms(wb_pair)                       #hide
@test rms(KE_resid .+ APE_resid) < 0.02 * rms(εₖ_pair)                      #hide
## and εₐ takes both signs over the run, which is what the discussion below rests on            #hide
@test minimum(εₐ_pair) < 0 < maximum(εₐ_pair);                           #hide

fig4 = Figure(size = (900, 760))

budget_kwargs = (xlabel = "Time", ylabel = "Rate", height = 190, width = 560)

ax_KE_bud = Axis(fig4[1, 1]; title = "Volume-integrated KE budget", budget_kwargs...)
lines!(ax_KE_bud, t_pair, -dKEdt,   label = "-d(∫KE)/dt")
lines!(ax_KE_bud, t_pair, wbᵣ_pair, label = "∫wbᵣ dV")
lines!(ax_KE_bud, t_pair, -εₖ_pair, label = "-∫εₖ dV")
lines!(ax_KE_bud, t_pair, KE_resid; label = "residual", color = :black, linestyle = :dash)
Legend(fig4[1, 2], ax_KE_bud; labelsize = 12, framevisible = false)

ax_APE_bud = Axis(fig4[2, 1]; title = "Volume-integrated APE budget", budget_kwargs...)
lines!(ax_APE_bud, t_pair, -dAPEdt,   label = "-d(∫eₐ)/dt")
lines!(ax_APE_bud, t_pair, -wbᵣ_pair, label = "-∫wbᵣ dV")
lines!(ax_APE_bud, t_pair, -εₐ_pair, label = "-∫εₐ dV")
lines!(ax_APE_bud, t_pair, APE_resid; label = "residual", color = :black, linestyle = :dash)
Legend(fig4[2, 2], ax_APE_bud; labelsize = 12, framevisible = false)

# `εₐ` is small enough next to the exchange term that it sits on top of the axis in the panel above, so
# the third panel drops the two large terms and keeps only the two dissipations and the residuals. This
# is the panel that says the APE budget is actually closed: the residual is not merely small next to
# `d(∫eₐ)/dt`, it is small next to `∫εₐ dV`, the smallest term in the budget.

ax_small = Axis(fig4[3, 1]; title = "The small terms, magnified", budget_kwargs...)
lines!(ax_small, t_pair, -εₐ_pair, label = "-∫εₐ dV", color = Cycled(3))
lines!(ax_small, t_pair, APE_resid, label = "APE residual", color = :black, linestyle = :dash)
lines!(ax_small, t_pair, KE_resid,  label = "KE residual", color = :grey40, linestyle = :dot)
Legend(fig4[3, 2], ax_small; labelsize = 12, framevisible = false)

resize_to_layout!(fig4)
save("lock_release_budgets.png", fig4)
set_theme!() #hide
nothing #hide

# ![](lock_release_budgets.png)
#
# `∫wb dV` is the mirror line running through both panels: the collapse converts `eₐ` into `KE` while
# the fronts accelerate, and the seiche hands it back each time the flow runs up against an end wall.
# The two sinks are comparable in size, `εₐ` reaching about two thirds of `εₖ` at the peak of the mixing
# around `t = 6`, but they are not the same kind of quantity. `∫εₖ dV` only ever removes `KE`. `∫εₐ dV`
# is a sink for almost the whole run and briefly a small source around `t = 1` to `2`, which is what its
# definition allows: at `t = 0` the lock is uniform in the vertical, the reference state has nothing to
# diffuse along, and `εₐ` is the whole diapycnal mixing rate, but as the current lays the fluid out in
# layers the reference state's own diffusion catches up and for a moment overtakes it. A sign-definite
# quantity like the buoyancy variance dissipation `-2 qᵢ∂ᵢb`, which no down-gradient closure lets turn
# negative, cannot do that at all. That is the practical difference between `εₐ` and the variance
# dissipation it is easily confused with.
#
# Both residuals stay near zero. They do not vanish, and cannot: the discrete KE and `eₐ` equations are
# not derived from the discrete momentum and buoyancy equations the model steps, so the two sides agree
# only to the truncation error of a well-resolved flow. The same caveat applies to the KE budget of
# [the two-dimensional turbulence example](@ref two_d_turbulence_example), with one more source of
# discrepancy here: `Integral(eₐ)` of the local Holliday & McIntyre density samples the reference
# profile at the model's cell centers, and that midpoint quadrature is itself second order in `Δz`.

# ## Two ways of writing the exchange
#
# Both budgets above are closed with `∫wbᵣ dV`, and closing the KE one with `∫wb dV` instead would work
# just as well, since the two integrals agree. The reason is visible in the maps: what separates them is
# `w b✶(z)`, the work the flow does against the reference stratification, which is a flux divergence and
# integrates to nothing in a closed box.

wb_t  = FieldTimeSeries(filepath, "wb")
wbᵣ_t = FieldTimeSeries(filepath, "wbᵣ")

t_peak = t_e[argmax(abs.(wb_bud[idx1]))]   # the snapshot where the exchange is strongest
n_peak = argmin(abs.(times .- t_peak))

wb_map  = interior(wb_t[n_peak],  :, 1, :)
wbᵣ_map = interior(wbᵣ_t[n_peak], :, 1, :)
wb✶_map = wb_map .- wbᵣ_map                # `wb = wbᵣ + w b✶(z)` holds cell by cell

## the two conversions are genuinely different maps, so the agreement above is about the integral #hide
@test maximum(abs, wb✶_map) > 0.1 * maximum(abs, wb_map)                                          #hide
## while their difference is the piece that integrates away, which is what lets either one close  #hide
## the integrated budget: on a uniform grid that is the plain sum over the cells                   #hide
@test abs(sum(wb✶_map)) < 1e-8 * sum(abs, wb✶_map);                                               #hide

set_theme!(Theme(fontsize = 20)) #hide
fig5 = Figure(size = (900, 620))

## one range for all three panels, taken over all three: `wbᵣ` runs about twice as large as `wb` here,
## and scaling each panel to itself would hide that
lim = maximum(maximum(abs, conversion) for conversion in (wb_map, wbᵣ_map, wb✶_map))
conv_kwargs = (ylabel = "z", height = 150, aspect = DataAspect())
xs, zs = xnodes(grid, Center()), znodes(grid, Center())

ax_wb  = Axis(fig5[2, 1]; title = "wb,  the KE budget's production",   conv_kwargs...)
ax_wbᵣ = Axis(fig5[3, 1]; title = "wbᵣ,  the eₐ budget's release",    conv_kwargs...)
ax_dif = Axis(fig5[4, 1]; title = "wb - wbᵣ = w b✶(z)", xlabel = "x", conv_kwargs...)

hms = [heatmap!(ax, xs, zs, conversion; colormap = :balance, colorrange = (-lim, lim))
       for (ax, conversion) in zip((ax_wb, ax_wbᵣ, ax_dif), (wb_map, wbᵣ_map, wb✶_map))]
Colorbar(fig5[5, 1], hms[1]; vertical = false, height = 8)

Label(fig5[1, 1], "Conversion terms,  t = " * string(round(t_peak, digits = 1)),
      fontsize = 22, tellwidth = false)

resize_to_layout!(fig5)
save("lock_release_conversion.png", fig5)
set_theme!() #hide
nothing #hide

# ![](lock_release_conversion.png)
#
# `wbᵣ` is the local release of `eₐ`: it concentrates in the two intruding noses and falls away to
# nothing in the fluid the fronts have not reached yet, where each parcel still carries the buoyancy its
# own height calls for. `wb` does not, since it counts the reference stratification's own contribution
# as well, the banded pattern about `z = 0.5` in the bottom panel, and that contribution is large enough
# here to leave `wbᵣ` running at about twice the amplitude of `wb`. Only `wbᵣ` maps where available
# potential energy is actually being converted; only in the volume integral can `wb` stand in for it.
