# Subfilter available potential energy equation

The `SubFilterAvailablePotentialEnergyEquation` module provides diagnostics for the available potential
energy carried by the scales that a low-pass spatial filter ``\overline{(\,\cdot\,)}`` removes from the
buoyancy field. It is the potential-energy counterpart of the [Subfilter kinetic energy equation](@ref):
where that module splits the kinetic energy across the filter scale, this one splits the local
available potential energy ``e_a`` of
[the available potential energy equation](available_potential_energy_equation.md). The other half of
the split, the APE of the filtered buoyancy and its dissipation, lives in the
[Filtered available potential energy equation](@ref) and is re-exported here.

Both the full and the filtered buoyancy are measured against one shared reference profile
``(b^\star, z^\star)``, ordinarily the sorted state of the full buoyancy field. The subfilter
available potential energy is the filtered full APE minus the APE of the filtered buoyancy ``\bar b``,

```math
e_a^s = \overline{e_a(b, z)} - e_a(\bar b, z) ,
\qquad
e_a(b, z) = \int_{z^\star(b)}^{z} \left[b^\star(\tilde z) - b\right] \mathrm{d}\tilde z ,
```

computed by [`SubFilterAvailablePotentialEnergy`](@ref). The second term is the APE of the filtered
buoyancy, ``e_a^l = e_a(\bar b, z)``, computed by [`FilteredAvailablePotentialEnergy`](@ref). Since the
filter acts at fixed ``z``, ``e_a^s`` is also ``\overline{e_a(b, z^\star(\bar b))}``: the average over
the filter of the APE each parcel holds between its own reference height ``z^\star(b)`` and the one the
filtered buoyancy would assign it, ``z^\star(\bar b)``. Because ``e_a`` is convex in buoyancy, a filter
that averages at fixed ``z`` keeps ``e_a^s \geq 0`` pointwise, by Jensen's inequality.

Looking the filtered buoyancy up in a profile it did not itself produce is what
[`ProfileLookup`](@ref Oceanostics.BackgroundPotentialEnergyEquation.ProfileLookup) was built for, so
these diagnostics accept only that reference-height method: the default sorts the model's own buoyancy
into a [`VerticalSort`](@ref Oceanostics.BackgroundPotentialEnergyEquation.VerticalSort) column on
every `compute!`, a column you built yourself can be shared across diagnostics, and a profile given as
plain arrays holds the reference state fixed in time (which also makes the diagnostics sort-free).

!!! warning "Horizontal filters only"
    The filter has to act only in the horizontal directions, and every diagnostic on this page throws
    an `ArgumentError` for a filter with a vertical component. Such a filter mixes heights as well as
    buoyancies and needs extra steps, not currently implemented, to guarantee a non-negative APE.

## Deriving the subfilter available potential energy equation

Since ``e_a^s = \overline{e_a} - e_a^l``, its budget is the budget of ``\overline{e_a}`` minus that of
``e_a^l``. Filtering the
[full-field APE budget](@ref "Deriving the local available potential energy equation") (the filter
commutes with derivatives) and splitting the filtered advective flux into a resolved and a subfilter
part, ``\overline{u_i e_a} = \bar u_i \, \overline{e_a} + \tau(u_i, e_a)``, gives the budget of
``\overline{e_a}``,

```math
\partial_t \overline{e_a} = -\partial_i(\bar u_i \, \overline{e_a})
                            - \overline{w b_r}
                            - \partial_i\left[\tau(u_i, e_a) + \overline{\Upsilon q_i}\right]
                            - \overline{\varepsilon_a}
                            + \overline{R} ,
```

where ``\tau(f, g) = \overline{f g} - \bar f \, \bar g`` is the subfilter covariance of two fields, and
the buoyancy anomaly ``b_r = b - b^\star(z)``, the displacement potential ``\Upsilon = z^\star(b) - z``,
the diffusive buoyancy flux ``q_i``, the dissipation ``\varepsilon_a`` and the reference tendency ``R``
are those of the full-field budget. The
[filtered APE budget](@ref "Deriving the filtered available potential energy equation") is

```math
\partial_t e_a^l = -\partial_i(\bar u_i e_a^l)
                   - \bar w \, \overline{b_r}
                   - \Pi_a
                   - \partial_i\left[\Upsilon^l \left(\tau(u_i, b) + \bar q_i\right)\right]
                   - \varepsilon_a^l
                   + R^l ,
```

and subtracting it from the budget of ``\overline{e_a}`` term by term gives the subfilter APE budget

```math
\partial_t e_a^s = \underbrace{-\partial_i(\bar u_i e_a^s)}_{\text{advection}}
                   \underbrace{-\,\tau(w, b_r)}_{\text{APE to KE conversion}}
                   + \underbrace{\Pi_a}_{\text{cross-scale flux}}
                   \underbrace{-\,\partial_i F^s_i}_{\text{subfilter transport}}
                   \underbrace{-\,\varepsilon_a^s}_{\text{dissipation}}
                   + \underbrace{R^s}_{\text{reference tendency}} ,
```

with

```math
\tau(w, b_r) = \overline{w b_r} - \bar w \, \overline{b_r} ,
\qquad
F^s_i = \tau(u_i, e_a) + \overline{\Upsilon q_i} - \Upsilon^l \left(\tau(u_i, b) + \bar q_i\right) ,
\qquad
\varepsilon_a^s = \overline{\varepsilon_a} - \varepsilon_a^l ,
\qquad
R^s = \overline{R} - R^l .
```

The cross-scale flux ``\Pi_a = -\tau(u_i, b) \, \partial_i \Upsilon^l`` appears with the opposite sign
in the filtered budget, so it moves APE between the filtered and the subfilter scales, from the former
to the latter where ``\Pi_a > 0``. The conversion ``\tau(w, b_r)`` likewise appears with the opposite
sign in the [Subfilter kinetic energy equation](@ref), which makes it a reversible exchange between the
subfilter APE and the subfilter kinetic energy. With the filtered conversion ``\bar w \, \overline{b_r}``
([`FilteredAvailablePotentialToKineticEnergyConversion`](@ref Oceanostics.FilteredAvailablePotentialEnergyEquation.FilteredAvailablePotentialToKineticEnergyConversion))
it adds up to ``\overline{w b_r}``. For the horizontal filters accepted here ``b^\star`` depends on ``z``
alone, so ``\overline{b_r} = \bar b - b^\star(z)``, the reference profile cancels between the two terms
of ``\tau(w, b_r)``, and the conversion is the subfilter buoyancy flux,
``\tau(w, b_r) = \tau(w, b) = \overline{w b} - \bar w \, \bar b``.

The dissipation ``\varepsilon_a^s`` is the filtered full-field dissipation
``\varepsilon_a = -q_i \, \partial_i \Upsilon``
([`AvailablePotentialEnergyDissipationRate`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate))
minus the dissipation of the filtered state, ``\varepsilon_a^l = -\bar q_i \, \partial_i \Upsilon^l``
([`FilteredAvailablePotentialEnergyDissipationRate`](@ref)), where ``\bar q_i`` is the closure's
diffusive buoyancy flux low-pass filtered and ``\Upsilon^l = z^\star(\bar b) - z`` is the displacement
potential of the filtered buoyancy. Filtering the flux, rather than recomputing it from ``\bar b``, is
the same choice the [Filtered kinetic energy equation](@ref) makes for the viscous flux; the
[Filtered available potential energy equation](@ref) has the details.

The flux ``F^s_i`` gathers the subfilter advection of ``e_a``, the filtered diffusive transport of the
full field and, with the opposite sign, the transport of the filtered budget. Its divergence
redistributes ``e_a^s`` in space and carries no APE across the filter scale. ``R^s`` is the subfilter
part of the reference tendency. It satisfies ``\int R^s \, \mathrm{d}V = -\int R^l \, \mathrm{d}V``, as
the [Filtered available potential energy equation](@ref) explains, and vanishes identically when the
reference profile is held fixed in time.

## Terms and diagnostics

Four of the quantities above have diagnostics: three are defined in this module, and the cross-scale
flux is defined in the [Filtered available potential energy equation](@ref) and re-exported here. The
advection, the subfilter transport and the reference tendency have none.

| Quantity | Expression | Diagnostic |
|:---|:---|:---|
| Subfilter available potential energy | ``e_a^s = \overline{e_a(b, z)} - e_a(\bar b, z)`` | [`SubFilterAvailablePotentialEnergy`](@ref) |
| Advection | ``\partial_i(\bar u_i e_a^s)`` | not implemented |
| APE to KE conversion | ``\tau(w, b_r) = \overline{w b_r} - \bar w \, \overline{b_r}`` | [`SubFilterAvailablePotentialToKineticEnergyConversion`](@ref) |
| Cross-scale flux | ``\Pi_a = -\tau(u_i, b) \, \partial_i \Upsilon^l`` | [`AvailablePotentialEnergyCrossScaleFlux`](@ref Oceanostics.FilteredAvailablePotentialEnergyEquation.AvailablePotentialEnergyCrossScaleFlux) |
| Subfilter transport | ``\partial_i F^s_i`` | not implemented |
| Dissipation | ``\varepsilon_a^s = \overline{\varepsilon_a} - \varepsilon_a^l`` | [`SubFilterAvailablePotentialEnergyDissipationRate`](@ref) |
| Reference tendency | ``R^s = \overline{R} - R^l`` | not implemented |

``\varepsilon_a^s`` also answers to `DissipationRate`. Like its full-field and filtered namesakes, the
alias is scoped to its module: `using Oceanostics.SubFilterAvailablePotentialEnergyEquation` brings it
in, `using Oceanostics` does not.

## Summary of ``e_a^s`` equation terms

```@docs
Oceanostics.SubFilterAvailablePotentialEnergyEquation.SubFilterAvailablePotentialEnergy
Oceanostics.SubFilterAvailablePotentialEnergyEquation.SubFilterAvailablePotentialToKineticEnergyConversion
Oceanostics.SubFilterAvailablePotentialEnergyEquation.AvailablePotentialEnergyCrossScaleFlux
Oceanostics.SubFilterAvailablePotentialEnergyEquation.SubFilterAvailablePotentialEnergyDissipationRate
```
