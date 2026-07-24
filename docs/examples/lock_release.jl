# # [Lock release and the sorted reference state](@id lock_release_example)
#
# In this example we run a two-dimensional lock release simulation and use it to watch the *sorted reference
# state*, available potential energy and kinetic energy evolve. Along the way we build the reference profile
# with each of the four methods Oceanostics offers and compare what they do and do not agree on.
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
# profile all the way from a step to something smooth. The grid is isotropic at `Δ = H/128`:

Nz = 128
grid = RectilinearGrid(size = (4Nz, Nz), x = (-Lx/2, Lx/2), z = (0, H), topology = (Bounded, Flat, Bounded))

model = NonhydrostaticModel(grid; timestepper = :RungeKutta3,
                            advection = WENO(order=5), # Diffusive and bounded scheme
                            buoyancy = BuoyancyTracer(), tracers = :b)

# The lock itself: buoyant fluid on the right, dense fluid on the left, separated by an interface a
# couple of cells thick. Smoothing the step over `δ` keeps the initial condition off the grid scale
# without changing the fact that almost every cell starts at one of two buoyancies:

δ = 2 * minimum_xspacing(grid)          # interface thickness, two cells
lock_release(x, z) = (Δb / 2) * tanh(x / δ)
set!(model, b = lock_release)

simulation = Simulation(model, Δt = 0.1 * minimum_xspacing(grid) / U, stop_time = 20)
conjure_time_step_wizard!(simulation, IterationInterval(5), cfl = 0.7)

using Oceanostics
progress = ProgressMessengers.BasicMessenger()
simulation.callbacks[:progress] = Callback(progress, IterationInterval(500))

# ## Diagnostics
#
# We build one `z★` per method and write all of them, so the reference state comes out of the run
# rather than being rebuilt afterwards. They are ordinary `Field`s that re-sort themselves whenever
# they are computed, so they can be passed to an output writer.
#
# What each method gives you to write differs, and that difference is the same one the figures below
# show. The three model-grid methods produce a *map* of the reference height, one value per cell, on the
# model grid. [`VerticalSort`](@ref) instead produces the sorted column itself, so its `z★` and
# the buoyancy that goes with it are already the profile, in order, and need no post-processing at all.
#
# Given that the referece height calculation is nonnlocal, it is generally a computationally-heavy operation
# and its cost grows faster than the number of cells: see [Computational cost per method](@ref) for how different
# methods compare and how they scale.

b = model.tracers.b

z★_ranked    = reference_height(model, method = ThreeDimensionalSort())
z★_heaviside = reference_height(model, method = HeavisideIntegral())
z★_lookup    = reference_height(model, method = ProfileLookup())
z★_column    = reference_height(model, method = VerticalSort())
b✶_column    = reference_buoyancy(z★_column)

# Both background and available potential energies are built from the same reference height, so we share one rather than letting each
# diagnostic sort the domain for itself. Note that `Eₐ` here is the *local* available potential energy
# (as defined by Holliday & McIntyre (1981)) and it should always be non-negative

#
# We build one `Eₐ` per method. Three of the four answer on the model grid and give a map over the
# domain, which is what the animation below compares. [`VerticalSort`](@ref) answers on the sorted
# column, so its `Eₐ` is ordered by rank rather than by position and is not a map of the flow; it is
# written anyway because the volume integral comes out the same whichever method builds the reference
# state. [`ProfileLookup`](@ref) is that same column read back onto the model grid, matching each cell
# to the profile through its buoyancy rather than through where it came from.

APE_ranked    = AvailablePotentialEnergy(model, z★_ranked)
APE_heaviside = AvailablePotentialEnergy(model, z★_heaviside)
APE_lookup    = AvailablePotentialEnergy(model, z★_lookup)
APE_column    = AvailablePotentialEnergy(model, z★_column)
KE            = KineticEnergy(model)

∫BPE = Integral(BackgroundPotentialEnergy(model, z★_ranked))
∫KE  = Integral(KE)

∫APE           = Integral(APE_ranked)
∫APE_heaviside = Integral(APE_heaviside)
∫APE_lookup    = Integral(APE_lookup)
∫APE_column    = Integral(APE_column)

using NCDatasets
filename = "lock_release"

# A single `NetCDFWriter` copes with the two grids: the model-grid fields are written against `x` and
# `z`, and the column's against its own `N`-cell vertical axis.

outputs = (; b, KE, APE_ranked, APE_heaviside, APE_lookup, APE_column,
             z★_ranked, z★_heaviside, z★_lookup, z★_column, b✶_column,
             ∫BPE, ∫KE, ∫APE, ∫APE_heaviside, ∫APE_lookup, ∫APE_column)

simulation.output_writers[:fields] = NetCDFWriter(model, outputs,
                                                  filename = joinpath(@__DIR__, filename),
                                                  schedule = TimeInterval(0.5),
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
# two model-grid methods, `z★` and `b` are both maps over the cells, so pairing them and ordering by
# `z★` recovers the same profile; that reordering is all the "post-processing" amounts to.

using Oceananigans.Fields: interior

filepath = simulation.output_writers[:fields].filepath
b_t = FieldTimeSeries(filepath, "b")     # for the movie below

ds = NCDataset(filepath)
times = ds["time"][:]
B  = ds["b"][:, :, :]                    # (x, z, time); y is Flat, so it is dropped
Z3 = ds["z★_ranked"][:, :, :]
ZH = ds["z★_heaviside"][:, :, :]
Z1 = ds["z★_column"][:, :, :]            # (1, N, time): the column keeps the model's Flat y, dropped here
B1 = ds["b✶_column"][:, :, :]
close(ds)

## pair a reference-height map with the buoyancy map and order by z★
mapped_profile(Z, n) = (h = vec(Float64.(Z[:, :, n])); p = sortperm(h); (vec(Float64.(B[:, :, n]))[p], h[p]))

## the column is already ordered, so it is read straight off
column_profile(n) = (vec(Float64.(B1[:, :, n])), vec(Float64.(Z1[:, :, n])))

snapshot_times = [0, 5, 10, 20]
snapshots = [argmin(abs.(times .- t)) for t in snapshot_times]

methods = ("ThreeDimensionalSort" => n -> mapped_profile(Z3, n),
           "HeavisideIntegral"    => n -> mapped_profile(ZH, n),
           "VerticalSort"   => column_profile)

## `profiles[name][k]` is the `(b✶, z★)` pair for method `name` at the `k`-th snapshot
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
z★_3d_0,  z★_hv_0 = profiles["ThreeDimensionalSort"][1][2], profiles["HeavisideIntegral"][1][2]   #hide
b✶_3d, z★_3d = profiles["ThreeDimensionalSort"][end]                                      #hide
b✶_hv, z★_hv = profiles["HeavisideIntegral"][end]                                         #hide
b✶_1d, z★_1d = profiles["VerticalSort"][end]                                        #hide
## the two model-grid methods differ by a quarter of the depth while the lock is intact   #hide
@test maximum(abs, z★_hv_0 .- z★_3d_0) > 0.2H                                             #hide
## and agree to the grid scale once mixing has made the field continuous                  #hide
@test maximum(abs, z★_hv .- z★_3d) < H / Nz                                               #hide
## the column carries exactly what the ranked sort assigns, by construction               #hide
@test maximum(abs, z★_3d .- z★_1d) < 1e-12                                                #hide
@test maximum(abs, b✶_3d .- b✶_1d) < 1e-12                                                #hide
## and the reference profile is, by construction, monotonic                               #hide
@test issorted(b✶_1d)                                                                     #hide
## The three methods describe one reference state, so although their `z★` maps differ wherever    #hide
## cells are tied, every buoyancy-weighted integral of `z★` has to agree — that integral is `BPE`, #hide
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
## a fifth of the depth out at `t = 0` and converges to the others as mixing removes the ties.     #hide
Δ_hv(n) = maximum(abs, vec(Float64.(ZH[:, :, n])) .- vec(Float64.(Z3[:, :, n])))          #hide
@test Δ_hv(snapshots[1])   > 0.2H                                                         #hide
@test Δ_hv(snapshots[end]) < H / Nz;                                                      #hide

# The profile gets a figure of its own: one panel per method, each showing `b★(z★)` at the four times
# above. It starts as a step, two blocks of uniform buoyancy stacked one on the other, and mixing
# erodes it into a smooth stratification.

using CairoMakie

set_theme!(Theme(fontsize = 20))
fig = Figure();

colors = cgrad(:viridis, length(snapshots); categorical = true)

for (m, (name, _)) in enumerate(methods)
    row, col = fldmod1(m, 2)   # fill a 2×2 grid row by row
    local ax = Axis(fig[row, col]; xlabel = "b✶", title = name, width = 200, height = 280,
                    ylabel = col == 1 ? "z★" : "", yticklabelsvisible = col == 1, titlesize = 15)
    ylims!(ax, 0, H)
    for (s, n) in enumerate(snapshots)
        b✶, z★ = profiles[name][s]
        lines!(ax, b✶, z★; color = colors[s], linewidth = 2, label = "t = $(round(times[n], digits=1))")
    end
    m == 1 && axislegend(ax; position = :lt, labelsize = 11)
end

# The fourth panel puts the three side by side at `t = 0`, where they disagree the most.
# [`HeavisideIntegral`](@ref) is drawn as markers rather than a line because its `z★` takes only as many
# distinct values as there are distinct buoyancies, which is a few dozen here against 65536 cells.

row0, col0 = fldmod1(length(methods) + 1, 2)
ax0 = Axis(fig[row0, col0]; xlabel = "b✶", title = "t = 0, all three", width = 200, height = 280,
           ylabel = col0 == 1 ? "z★" : "", yticklabelsvisible = col0 == 1, titlesize = 15)
ylims!(ax0, 0, H)

b✶_r, z★_r = profiles["ThreeDimensionalSort"][1]
b✶_c, z★_c = profiles["VerticalSort"][1]
b✶_h, z★_h = profiles["HeavisideIntegral"][1]

lines!(ax0, b✶_r, z★_r; linewidth = 5, color = (:steelblue, 0.9), label = "ThreeDimensionalSort")
lines!(ax0, b✶_c, z★_c; linewidth = 2, linestyle = :dash, color = :black, label = "VerticalSort")
scatter!(ax0, b✶_h, z★_h; markersize = 9, color = :crimson, label = "HeavisideIntegral")
axislegend(ax0; position = :lt, labelsize = 9)

# The three method panels are identical except while the lock is still intact, and that difference is
# informative rather than an error. At `t = 0` almost every cell is tied with thousands of others at
# one of two buoyancies, and the methods place tied cells differently.
# [`ThreeDimensionalSort`](@ref) and [`VerticalSort`](@ref) give each cell its own slot in the
# stack, so they draw the true step spanning the full depth. [`HeavisideIntegral`](@ref) instead
# collapses each buoyancy class onto the mid-height of the layer it fills, which is what makes `z★` a
# function of buoyancy alone and a clean field to map, but leaves it unable to represent a step as a
# profile: its `z★` only ever reaches the mid-heights of the two blocks, about a quarter of the depth
# in from each boundary. Once mixing has made the buoyancy field continuous the ties vanish and all
# three agree to within a grid cell.

resize_to_layout!(fig)
save("lock_release_profiles.png", fig)
set_theme!() #hide
nothing #hide

# ![](lock_release_profiles.png)

# ## Flow animation and local energies
#
# The movie sets the flow beside the energy it carries: buoyancy, kinetic energy, and the local `Eₐ`
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
ax_E3 = Axis(fig3[6, 1];  title = "Eₐ,  ThreeDimensionalSort",       panel_kwargs...)
ax_EH = Axis(fig3[8, 1];  title = "Eₐ,  HeavisideIntegral",          panel_kwargs...)
ax_EL = Axis(fig3[10, 1]; title = "Eₐ,  ProfileLookup", xlabel = "x", panel_kwargs...)

bₙ  = @lift b_t[$n]
KEₙ = @lift KE_t[$n]
E3ₙ = @lift APE3_t[$n]
EHₙ = @lift APEH_t[$n]
ELₙ = @lift APEL_t[$n]

## `Eₐ` and the kinetic energy are both sign-definite, so they get one-sided ranges set from their own
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
# `Eₐ` drains from the lock as the fronts accelerate and refills wherever the seiche lifts dense fluid
# back above its reference height. Being the local form, it is non-negative everywhere. The three `Eₐ`
# panels are identical: the methods differ only in where inside a tied run they place `z★`, and `Eₐ`
# cannot see that choice.

# ## Energetics
#
# The same three energies, now volume integrated. Their sum (dashed) is the total, which no exchange
# among them can alter.

ds = NCDataset(simulation.output_writers[:fields].filepath)
t_e     = ds["time"][:]
BPE_int = ds["∫BPE"][:]
APE_int = ds["∫APE"][:]
KE_int  = ds["∫KE"][:]
## all four methods integrate to the same Eₐ, however they place cells of equal buoyancy  #hide
@test ds["∫APE_heaviside"][:] ≈ APE_int rtol=1e-8                                          #hide
@test ds["∫APE_lookup"][:]    ≈ APE_int rtol=1e-8                                          #hide
@test ds["∫APE_column"][:]    ≈ APE_int rtol=1e-8                                          #hide
close(ds)

total_int = KE_int .+ APE_int .+ BPE_int

@test APE_int[1] > 0                                        # a lock is pure available PE     #hide
@test minimum(APE_int) < 0.05 * APE_int[1]                  # the collapse nearly empties it  #hide
@test all(APE_int .≥ -1e-8)                                 # and it is never negative        #hide
@test KE_int[1] < 1e-8 * APE_int[1]                         # the lock starts at rest         #hide
@test BPE_int[end] > BPE_int[1]                             # mixing raised the background    #hide
@test minimum(diff(BPE_int)) > -1e-6 * maximum(abs, BPE_int) # and only ever raised it        #hide
## the total is only ever dissipated, never created                                           #hide
@test maximum(diff(total_int)) < 1e-6 * abs(total_int[1] - total_int[end]);                   #hide

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
# last. The model has no explicit dissipation, so the dashed total drifts down only through the
# advection scheme's implicit dissipation.
