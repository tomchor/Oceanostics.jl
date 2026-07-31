# Background potential energy equation

The `BackgroundPotentialEnergyEquation` module builds the **reference state**: the arrangement of the
buoyancy field with the least potential energy reachable adiabatically. Following
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X), rearrange the field with every
parcel sorted by density and the densest fluid at the bottom. The height a parcel ends up at is its
reference height ``z^\star``, and the potential energy of that state is the background (or reference)
potential energy,

```math
e_b = -b z^\star = \frac{g \rho}{\rho_0} z^\star .
```

``e_b`` is the share of the potential energy ``e_p = -bz`` of
[the potential energy equation](potential_energy_equation.md) that the flow *cannot* release. What is
left over is the [available potential energy](available_potential_energy_equation.md), so the two
modules are the two halves of one split and share the reference state built here: `reference_height`,
`reference_buoyancy` and the methods below are exported by both.

## The background potential energy equation

``e_p`` is ``-z`` times the buoyancy with ``z`` fixed, so its equation follows from the buoyancy
equation cell by cell. ``e_b`` is ``-z^\star`` times the same buoyancy, and ``z^\star`` is a functional
of the *whole* field: it comes out of a sort, so a change anywhere in the domain can move it. There is
no local ``e_b`` equation in the sense [the potential energy page](potential_energy_equation.md) derives
one. What follows is the equation for the volume integral, which is what ``E_b`` is used for anyway.

Two facts carry the derivation. The first is that ``E_b`` depends on the buoyancy field only through its
*distribution*, the volume of fluid holding each buoyancy: sorting discards where every parcel sits and
keeps only how much fluid is at each value. The second is that advection moves parcels around without
changing the buoyancy they carry, so it leaves that distribution untouched. Adiabatic motion therefore
cannot change ``E_b`` at all, however violent, and only the diffusive part of ``\partial_t b`` survives
([Winters et al., 1995](https://doi.org/10.1017/S002211209500125X)):

```math
\frac{d}{dt}\int e_b \, \mathrm{d}V
    = -\int z^\star \, \partial_t b \big|_\mathrm{diffusive} \, \mathrm{d}V
    = \int z^\star \, \partial_j q_j \, \mathrm{d}V ,
```

with ``q_j`` the closure's diffusive flux of buoyancy, in the convention
[the potential energy page](potential_energy_equation.md) sets out. Pulling ``z^\star`` inside the
derivative splits that the same way, except that ``z^\star`` varies through ``b`` rather than through
position, so ``\partial_j z^\star = (\partial z^\star/\partial b)\,\partial_j b``:

```math
z^\star \partial_j q_j = \partial_j (z^\star q_j) - q_j \partial_j z^\star
                       = \partial_j (z^\star q_j)
                         + \kappa \frac{\partial z^\star}{\partial b} \left|\nabla b\right|^2 .
```

The divergence integrates to a flux through the boundary, which vanishes for insulating walls, leaving
one term:

```math
\frac{d}{dt}\int e_b \, \mathrm{d}V
    = \int \kappa \frac{\partial z^\star}{\partial b} \left|\nabla b\right|^2 \mathrm{d}V
    \equiv \phi_d \ge 0 ,
```

the diapycnal mixing rate of Winters et al. It is non-negative because ``z^\star`` rises with ``b``, so
mixing across buoyancy surfaces can only raise the background potential energy. That one-way property is
what makes ``\int e_b \, \mathrm{d}V`` a mixing measure, and it is also why it measures a scheme's
*spurious* mixing: the continuous equations say advection cannot move ``E_b``, so whatever motion of it
survives in a simulation with no explicit diffusion came from the advection scheme.

``\phi_d`` has no diagnostic of its own, and needs none, because it splits into two that do. Writing it
as ``\kappa \nabla b \cdot \nabla z^\star`` and substituting ``z^\star = z + \Upsilon``, with
``\Upsilon`` the [buoyancy displacement potential](available_potential_energy_equation.md),

```math
\phi_d = \kappa \nabla b \cdot \nabla z^\star
        = \underbrace{\kappa \nabla b \cdot \nabla \Upsilon}_{\varepsilon_A}
        + \underbrace{\kappa \, \partial b / \partial z}_{\Phi} ,
```

the [APE dissipation rate](available_potential_energy_equation.md) and the
[diffusive buoyancy flux](potential_energy_equation.md#Diffusive-buoyancy-flux). The two split the
mixing by what it costs the flow: ``\varepsilon_A`` is the part paid for out of available potential
energy, and ``\Phi`` is the part diffusion does to the reference state on its own, which carries no
available energy with it. So

```math
\frac{d}{dt}\int e_b \, \mathrm{d}V = \int \left(\varepsilon_A + \Phi\right) \mathrm{d}V .
```

Neither term is sign-definite on its own; their sum is, which is the sharpest check available on either
of them.

### Terms and what is implemented

| Quantity | Expression | Diagnostic |
|:---|:---|:---|
| Background potential energy | ``e_b = -b z^\star`` | [`BackgroundPotentialEnergy`](@ref Oceanostics.BackgroundPotentialEnergyEquation.BackgroundPotentialEnergy) |
| Reference height | ``z^\star`` | [`reference_height`](@ref Oceanostics.BackgroundPotentialEnergyEquation.reference_height) |
| Advection | vanishes identically | not applicable |
| Diffusive transport | ``\partial_j(z^\star q_j)`` | not implemented |
| Diapycnal mixing rate | ``\phi_d = \kappa \nabla b \cdot \nabla z^\star = \varepsilon_A + \Phi`` | the sum of the two below |
| APE dissipation rate | ``\varepsilon_A = \kappa \nabla b \cdot \nabla \Upsilon`` | [`AvailablePotentialEnergyDissipationRate`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate) |
| Diffusive buoyancy flux | ``\Phi = \kappa \, \partial b / \partial z`` | [`DiffusiveBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveBuoyancyFlux) |

!!! note "``\partial z^\star / \partial b`` needs ``z^\star`` to be a function of ``b``"
    The chain rule step above assumes the reference height depends on position only through the
    buoyancy, which is exact in the continuum but a choice at finite resolution. It holds cell by cell
    for [`HeavisideIntegral`](@ref) and [`ProfileLookup`](@ref), which give every cell of a tied run the
    same ``z^\star``. [`ThreeDimensionalSort`](@ref) hands tied cells consecutive slots instead, so
    ``z^\star`` spreads over the depth the run fills. That spread is the volume-weighted mean of what
    the other two assign, so every volume integral on this page is unaffected; only pointwise use of
    ``\partial z^\star/\partial b`` is.

## The reference state

height ``z^\star`` is the height a parcel would occupy in a state of minimum potential energy
that can be reached by adiabatic rearrangement flow parcels, and it
is computed by [`reference_height`](@ref). Rearrangement is a nonlocal
operation, so, unlike most other diagnostics in Oceanostics, this one is not a pointwise kernel.
It is rearranged on each `compute!`, so writing it (or anything built on it) out during a simulation
tracks the evolving flow.

Organizing the buoyancy against the reference height it was assigned gives the reference profile
``b^\star(z^\star)``: the stratification the flow would have if all of its available potential energy
were released. The [Lock release](@ref lock_release_example) example follows that profile through a
gravity current, from the step it starts as to the smooth stratification mixing leaves behind, and
builds it with each of the four methods below so their costs and their differences can be compared
directly.

`method` selects one of four strategies to calculate the reference state. All
`method`s produce the same reference state in the continuous limit and the same ``\int e_a \, \mathrm{d}V``.
Mainly what differs is how cells of *equal* buoyancy are placed, and what grid the answer lands on.
Here is a brief summary of the four, with more detail in the docstring of each:

  - [`ThreeDimensionalSort`](@ref) (the default) ranks the cells and gives each one the height of its
    own slot in the sorted state, on the model grid. Tied cells take consecutive slots rather than a
    shared height, so ``z^\star`` spreads over a grid cell wherever the stratification is horizontally
    uniform. Crucially, this method may leave small horizontal buoyancy gradients in the reference
    state, which goes against the idea of a reference state being horizontally uniform.

  - [`VerticalSort`](@ref) is similar to [`ThreeDimensionalSort`](@ref) but returns the cells
    reorganized into a single column. To achieve this the cells are flattened such that their volume
    remains the same, but their horizontal area matches the domain's horizontal area. This has the
    advantage that the resulting reference state (correctly) has no horizontal structure. The downside
    is that the results land on a different grid, namely a ``1 \times 1 \times N`` grid whose
    ``N = N_x N_y N_z`` cells span the domain's full horizontal area.

  - [`ProfileLookup`](@ref) gives each cell the height of the slot whose buoyancy matches its own,
    found by binary search into a sorted profile. Cells are matched by value rather than by identity,
    so it is the one method whose profile need not have come from the field being diagnosed. It
    reproduces on the model grid what the [`VerticalSort`](@ref) column holds, without calling that
    method to do it.

  - [`HeavisideIntegral`](@ref) is Eq. (11) of Winters et al. (1995) verbatim, with the Heaviside step
    function ``H`` taking the value ``1/2`` where the two densities are equal. That half-weight gives
    every cell of a given buoyancy the same ``z^\star``, the mid-height of the layer that buoyancy
    class fills in the sorted column, which makes ``z^\star`` a function of buoyancy alone and constant
    on isopycnals. Because it builds ``z^\star`` from a volume fraction rather than by stacking cells
    into a column, it is also the only method that works on a **stretched grid** (one with non-uniform
    cell volumes).

Written out, that last one is

```math
z^\star(\boldsymbol{x}) = z_\mathrm{bottom} + \frac{1}{A} \int H\!\left(\rho(\boldsymbol{x}') - \rho(\boldsymbol{x})\right) \mathrm{d}V' ,
```

with ``A`` the domain's horizontal area. Taken literally that is a double integral, a sweep over the
whole domain for every cell, costing ``\mathcal{O}(N^2)``. Sorting the cells by buoyancy first collapses
it to a cumulative sum: with the cells ordered densest first and ``V_n = \sum_{p \le n} \Delta V_p`` the
running total of their volumes, every cell of the tied run spanning ranks ``p`` through ``q`` takes

```math
z^\star = z_\mathrm{bottom} + \frac{V_{p-1} + V_q}{2A} ,
```

where ``V_{p-1}`` is the volume strictly denser than the parcel and ``V_q`` the volume no lighter than
it, the pair that ``H = 1/2`` averages at equality. A single `cumsum` over the sorted volumes therefore
serves every cell, which brings the cost back down to the ``\mathcal{O}(N \log N)`` of the sort.

## Computational cost per method

Sorting couples every cell in the domain to every other one, so unlike the pointwise diagnostics its
cost is not linear in the number of cells. All four methods pay for the same `sortperm!`, which is
``\mathcal{O}(N \log N)``; what separates them is the work done around it.
[`HeavisideIntegral`](@ref) makes extra passes over the sorted cells to find the tied runs,
[`ProfileLookup`](@ref) adds a binary search per cell, and [`VerticalSort`](@ref) carries the
buoyancy and the original heights into the column.

To measure that, build the same synthetic field — a linear stratification plus noise, so no two cells
are tied and the sort does its full work — on four grids spanning two decades in cell count, and time
one `compute!` of ``z^\star`` on each.

```@example ape_timing
using Oceananigans, Oceanostics, CairoMakie, Random
using Oceananigans.Fields: CenterField, compute!

function noisy_field(N)
    grid = RectilinearGrid(size = (N, N), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded))
    b = CenterField(grid)
    Random.seed!(42)
    set!(b, reshape(znodes(grid, Center()), 1, 1, N) .+ 0.1 .* randn(N, 1, N))
    return b
end

Ns      = (32, 64, 128, 256)
cells   = collect(Ns .^ 2)   # a Vector, so Makie can plot it directly
methods = ("ThreeDimensionalSort" => ThreeDimensionalSort(),
           "HeavisideIntegral"    => HeavisideIntegral(),
           "ProfileLookup"        => ProfileLookup(),
           "VerticalSort"         => VerticalSort())

## best of several runs, after a warm-up so compilation stays out of the measurement
function time_sort(N, method; samples = 7)
    z✶ = reference_height(noisy_field(N); method)
    compute!(z✶)
    return minimum(@elapsed(compute!(z✶)) for _ in 1:samples)
end

timings = Dict(name => [1e3 * time_sort(N, method) for N in Ns] for (name, method) in methods)
nothing # hide
```

Plotted against cell count on log axes, alongside an ``N \log N`` reference anchored at the largest
grid:

```@example ape_timing
fig = Figure(size = (620, 420))
ax  = Axis(fig[1, 1]; xlabel = "number of cells", ylabel = "time per sort (ms)",
           xscale = log10, yscale = log10, title = "Cost of building the reference state")

for (name, _) in methods
    scatterlines!(ax, cells, timings[name]; label = name, markersize = 10)
end

## N log N, scaled to meet the cheapest method at the largest grid
reference = cells .* log.(cells)
reference = reference ./ reference[end] .* timings["ThreeDimensionalSort"][end]
lines!(ax, cells, reference; color = :black, linestyle = :dash, label = "N log N")

axislegend(ax; position = :lt, labelsize = 11)
fig
```

The four curves run roughly parallel to the dashed reference, indicating that the sort is what sets the scaling.
If all you need are the volume integrals, the default is the cheapest route to them.

The absolute numbers are machine-dependent, and the docs are built with different optimisation
settings than a typical simulation, so read the *shape* of these curves rather than the values. The
spread between methods in particular is narrower here than on an optimised build.

## Background potential energy

```@docs
Oceanostics.BackgroundPotentialEnergyEquation.BackgroundPotentialEnergy
```

## Reference height and the methods that build it

```@docs
Oceanostics.BackgroundPotentialEnergyEquation.reference_height
Oceanostics.BackgroundPotentialEnergyEquation.reference_buoyancy
Oceanostics.BackgroundPotentialEnergyEquation.AbstractReferenceHeightMethod
Oceanostics.BackgroundPotentialEnergyEquation.ThreeDimensionalSort
Oceanostics.BackgroundPotentialEnergyEquation.HeavisideIntegral
Oceanostics.BackgroundPotentialEnergyEquation.ProfileLookup
Oceanostics.BackgroundPotentialEnergyEquation.VerticalSort
```
