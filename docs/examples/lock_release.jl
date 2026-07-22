# # [Lock release and the sorted reference state](@id lock_release_example)
#
# In this example we run a two-dimensional lock release and use it to watch the *sorted reference
# state* evolve. A lock release is about the sharpest test of a reference-state calculation there is:
# it starts as two blocks of uniform buoyancy sitting side by side, which is as far from a sorted state
# as a stratified fluid can get, and it ends well mixed. Along the way we build the reference profile
# with each of the three methods Oceanostics offers and time them against each other.
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
# `H` set the buoyancy velocity `U = √(Δb H) / 2`, which is the classic lock-release front speed, and
# from it the Reynolds number fixes the viscosity. The channel is four times as long as it is deep:

Δb = 1      # buoyancy jump across the lock
H  = 1      # channel depth
Lx = 4H     # channel length
Re = 1000   # Reynolds number
Pr = 1      # Prandtl number

U = √(Δb * H) / 2   # buoyancy velocity
ν = U * H / Re      # viscosity
κ = ν / Pr          # buoyancy diffusivity

# The domain is walled at both ends and at top and bottom, so the fronts eventually reflect and the
# channel fills with a mixed intermediate layer. That is what we want here: it drives the reference
# profile all the way from a step to something smooth. The grid is isotropic at `Δ = H/128`:

Nz = 128
grid = RectilinearGrid(size = (4Nz, Nz), x = (-Lx/2, Lx/2), z = (-H/2, H/2),
                       topology = (Bounded, Flat, Bounded))

model = NonhydrostaticModel(grid; timestepper = :RungeKutta3,
                            advection = Centered(order=4),
                            closure = ScalarDiffusivity(; ν, κ),
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
# they are computed, so an output writer can simply be handed them.
#
# What each method gives you to write differs, and that difference is the same one the figures below
# show. The two model-grid methods produce a *map* of the reference height, one value per cell, on the
# model grid. [`OneDimensionalSort`](@ref) instead produces the sorted column itself, so its `z★` and
# the buoyancy that goes with it are already the profile, in order, and need no post-processing at all.
#
# Building all three here costs three sorts per output rather than one. That is affordable at this
# resolution, but the sort is the one part of these diagnostics whose cost grows faster than the number
# of cells: see [What the three methods cost](@ref) for how the three compare and how they scale.

b = model.tracers.b

z★_ranked    = reference_height(model, method = ThreeDimensionalSort())
z★_heaviside = reference_height(model, method = HeavisideIntegral())
z★_column    = reference_height(model, method = OneDimensionalSort())
b✶_column    = reference_buoyancy(z★_column)

# Both energies are built from the same reference height, so we share one rather than letting each
# diagnostic sort the domain for itself.

∫E_b = Integral(BackgroundPotentialEnergy(model, z★_ranked))
∫E_a = Integral(AvailablePotentialEnergy(model, z★_ranked))

using NCDatasets
filename = "lock_release"

# A single `NetCDFWriter` copes with the two grids: the model-grid fields are written against `x` and
# `z`, and the column's against its own `N`-cell vertical axis.

simulation.output_writers[:fields] = NetCDFWriter(model,
                                                  (; b, z★_ranked, z★_heaviside, z★_column, b✶_column),
                                                  filename = joinpath(@__DIR__, filename),
                                                  schedule = TimeInterval(0.5),
                                                  overwrite_existing = true)

simulation.output_writers[:energies] = NetCDFWriter(model, (; ∫E_b, ∫E_a),
                                                    filename = joinpath(@__DIR__, filename * "_energies"),
                                                    schedule = TimeInterval(0.5),
                                                    overwrite_existing = true)

# ## Run the simulation

run!(simulation)

# ## The reference profile
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
b_t = FieldTimeSeries(filepath, "b")     # for the heatmaps below

ds = NCDataset(filepath)
times = ds["time"][:]
B  = ds["b"][:, :, :]                    # (x, z, time); y is Flat, so it is dropped
Z3 = ds["z★_ranked"][:, :, :]
ZH = ds["z★_heaviside"][:, :, :]
Z1 = ds["z★_column"][:, :, :]            # (1, N, time): the column keeps the model's Flat y, dropped here
B1 = ds["b✶_column"][:, :, :]
close(ds)

## pair a reference-height map with the buoyancy map and order by z★
mapped_profile(Z, n) = (h = vec(Float64.(Z[:, :, n])); p = sortperm(h);
                        (vec(Float64.(B[:, :, n]))[p], h[p]))

## the column is already ordered, so it is read straight off
column_profile(n) = (vec(Float64.(B1[:, :, n])), vec(Float64.(Z1[:, :, n])))

snapshot_times = [0, 5, 10, 20]
snapshots = [argmin(abs.(times .- t)) for t in snapshot_times]

methods = ("ThreeDimensionalSort" => n -> mapped_profile(Z3, n),
           "HeavisideIntegral"    => n -> mapped_profile(ZH, n),
           "OneDimensionalSort"   => column_profile)

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
b✶_1d, z★_1d = profiles["OneDimensionalSort"][end]                                        #hide
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
## cells are tied, every buoyancy-weighted integral of `z★` has to agree — that integral is `E_b`, #hide
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

# ## Animating the flow and its energy
#
# The panels above are snapshots; the exchange between the reservoirs is easier to follow as a movie.
# We animate three fields side by side: the buoyancy that drives the flow, the kinetic energy it
# produces, and the local available potential energy still stored in the density field. `Eₐ` is the one
# worth watching against the other two — it starts concentrated at the lock, is spent as the fronts run
# and the billows break, and refills wherever the seiche lifts dense fluid back up.

using CairoMakie

set_theme!(Theme(fontsize = 20))
fig = Figure(size = (1000, 760));

# The top row is the point of the example: the reference profile at a few times, one panel per method.
# It starts as a step, two blocks of uniform buoyancy stacked one on the other, and mixing erodes it
# into a smooth stratification.

colors = cgrad(:viridis, length(snapshots); categorical = true)

for (m, (name, _)) in enumerate(methods)
    local ax = Axis(fig[1, m]; xlabel = "b✶", title = name, width = 200, height = 280,
                    ylabel = m == 1 ? "z★" : "", yticklabelsvisible = m == 1, titlesize = 15)
    ylims!(ax, -H/2, H/2)
    for (s, n) in enumerate(snapshots)
        b✶, z★ = profiles[name][s]
        lines!(ax, b✶, z★; color = colors[s], linewidth = 2, label = "t = $(round(times[n], digits=1))")
    end
    m == 1 && axislegend(ax; position = :lt, labelsize = 11)
end

# The rightmost panel of that row puts the three side by side at `t = 0`, where they disagree the most.
# [`HeavisideIntegral`](@ref) is drawn as markers rather than a line because its `z★` takes only as many
# distinct values as there are distinct buoyancies, which is a few dozen here against 65536 cells.

ax0 = Axis(fig[1, length(methods) + 1]; xlabel = "b✶", title = "t = 0, all three",
           width = 200, height = 280, yticklabelsvisible = false, titlesize = 15)
ylims!(ax0, -H/2, H/2)

b✶_r, z★_r = profiles["ThreeDimensionalSort"][1]
b✶_c, z★_c = profiles["OneDimensionalSort"][1]
b✶_h, z★_h = profiles["HeavisideIntegral"][1]

lines!(ax0, b✶_r, z★_r; linewidth = 5, color = (:steelblue, 0.9), label = "ThreeDimensionalSort")
lines!(ax0, b✶_c, z★_c; linewidth = 2, linestyle = :dash, color = :black, label = "OneDimensionalSort")
scatter!(ax0, b✶_h, z★_h; markersize = 9, color = :crimson, label = "HeavisideIntegral")
axislegend(ax0; position = :lt, labelsize = 9)

# The three method panels are identical except while the lock is still intact, and that difference is
# informative rather than an error. At `t = 0` almost every cell is tied with thousands of others at
# one of two buoyancies, and the methods place tied cells differently.
# [`ThreeDimensionalSort`](@ref) and [`OneDimensionalSort`](@ref) give each cell its own slot in the
# stack, so they draw the true step spanning the full depth. [`HeavisideIntegral`](@ref) instead
# collapses each buoyancy class onto the mid-height of the layer it fills, which is what makes `z★` a
# function of buoyancy alone and a clean field to map, but leaves it unable to represent a step as a
# profile: its `z★` only ever reaches the mid-heights of the two blocks, about a quarter of the depth
# in from each boundary. Once mixing has made the buoyancy field continuous the ties vanish and all
# three agree to within a grid cell.

# The rows below show the flow those profiles come from: a vertical cross section of the buoyancy field
# at each of the same times. The dense fluid on the left runs right along the bottom, the buoyant fluid
# on the right runs left along the top, and the shear between them rolls up into billows that do the
# mixing.

n_cols = length(methods) + 1

for (row, n) in enumerate(snapshots)
    local ax = Axis(fig[row + 1, 1:n_cols]; ylabel = "z", width = 860, height = 215,
                    xlabel = row == length(snapshots) ? "x" : "",
                    xticklabelsvisible = row == length(snapshots))
    heatmap!(ax, b_t[n]; colormap = :balance, colorrange = (-Δb/2, Δb/2))
    text!(ax, -Lx/2 + 0.05, 0.30; text = "t = $(round(times[n], digits=1))", fontsize = 16)
end

Colorbar(fig[2:length(snapshots) + 1, n_cols + 1];
         colormap = :balance, limits = (-Δb/2, Δb/2), label = "b")

resize_to_layout!(fig)
save("lock_release_profiles.png", fig)
nothing #hide

# ![](lock_release_profiles.png)

# ## Energetics
#
# A lock release is the textbook case of the split this module computes. The initial state holds no
# kinetic energy and a great deal of available potential energy; the collapse converts `Eₐ` into motion,
# and the billows then mix irreversibly, which shows up as a rise in `E_b`. Because the channel is
# closed, the fronts reflect off the end walls and the whole box seiches, so `∫Eₐ dV` does not decay
# smoothly: it very nearly empties as the fronts pass each other, then refills as the sloshing carries
# fluid back up, with each cycle weaker than the last.

ds = NCDataset(simulation.output_writers[:energies].filepath)
t_e   = ds["time"][:]
E_b_t = ds["∫E_b"][:]
E_a_t = ds["∫E_a"][:]
close(ds)

@test E_a_t[1] > 0                                          # a lock is pure available PE     #hide
@test minimum(E_a_t) < 0.05 * E_a_t[1]                      # the collapse nearly empties it  #hide
@test E_b_t[end] > E_b_t[1]                                 # mixing raised the background    #hide
@test minimum(diff(E_b_t)) > -1e-6 * maximum(abs, E_b_t);   # and only ever raised it         #hide

fig2 = Figure(size = (700, 300))
ax = Axis(fig2[1, 1]; xlabel = "Time", ylabel = "Energy", title = "Lock-release energetics")
lines!(ax, t_e, E_a_t, label = "∫Eₐ dV")
lines!(ax, t_e, E_b_t .- E_b_t[1], label = "Δ∫E_b dV")
axislegend(ax; position = :rc, labelsize = 12)

save("lock_release_energetics.png", fig2)
nothing #hide

# ![](lock_release_energetics.png)
#
# The two curves separate the reversible part of the flow from the irreversible one. `∫Eₐ dV` swings up
# and down with the seiche, since sloshing lifts dense fluid back up and stores energy that the flow can
# still give back. `Δ∫E_b dV` only ever climbs: it is the running record of how much buoyancy has
# actually been mixed across density surfaces, and it is precisely the part of the reference profile's
# evolution that cannot be undone. By the end of the run it has absorbed a modest fraction of the
# available potential energy the lock started with, with the rest still sloshing between kinetic and
# available potential energy.
