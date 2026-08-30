# Sub-filter available potential energy equation

The `SubFilterAvailablePotentialEnergyEquation` module provides diagnostics for the available potential
energy carried by the scales that a low-pass spatial filter ``\overline{(\,\cdot\,)}`` removes from the
buoyancy field, following the filtered APE framework of
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879). It is the potential-energy
counterpart of the [Sub-filter kinetic energy equation](@ref): where that module splits the kinetic
energy across the filter scale, this one splits the *local* available potential energy ``e_a`` of
[the available potential energy equation](available_potential_energy_equation.md). The other half of
the split, the APE of the filtered buoyancy and its dissipation, lives in the
[Filtered available potential energy equation](@ref) and is re-exported here.

## The subfilter available potential energy

Both the full and the filtered buoyancy are measured against **one shared reference profile**
``(b^\star, z^\star)``, ordinarily the sorted state of the full buoyancy field. The subfilter
available potential energy is the filtered full APE minus the APE of the filtered buoyancy
``\bar b``,

```math
e_a^s = \overline{e_a(b, z)} - e_a(\bar b, z) ,
\qquad
e_a(b, z) = \int_{z^\star(b)}^{z} \left[b^\star(\tilde z) - b\right] \mathrm{d}\tilde z ,
```

computed by [`SubFilterAvailablePotentialEnergy`](@ref); ``e_a(\bar b, z)`` is
[`FilteredAvailablePotentialEnergy`](@ref). Looking the filtered buoyancy up in a profile
it did not itself produce is exactly what
[`ProfileLookup`](@ref Oceanostics.BackgroundPotentialEnergyEquation.ProfileLookup) was built for, so
these diagnostics accept only that reference-height method: the default sorts the model's own buoyancy
into a [`VerticalSort`](@ref Oceanostics.BackgroundPotentialEnergyEquation.VerticalSort) column on
every `compute!`, a column you built yourself can be shared across diagnostics, and a profile given as
plain arrays holds the reference state fixed in time (which also makes the diagnostics sort-free).

Because ``e_a`` is convex in buoyancy, a filter with no vertical component keeps ``e_a^s \geq 0``
pointwise, by Jensen's inequality; a filter that acts vertically mixes heights as well as buoyancies
and can produce locally negative values.

```@docs
Oceanostics.SubFilterAvailablePotentialEnergyEquation.SubFilterAvailablePotentialEnergy
```

## The subfilter available potential energy dissipation

The diffusive sink of the ``e_a^s`` budget is the subfilter APE dissipation rate

```math
\varepsilon_a^s = \overline{\varepsilon_a} - \varepsilon_a^l ,
\qquad
\varepsilon_a^l = -\bar q_i \, \partial_i \Upsilon^l ,
```

computed by [`SubFilterAvailablePotentialEnergyDissipationRate`](@ref): the filtered full-field
dissipation ``\varepsilon_a = -q_i \partial_i \Upsilon``
([`AvailablePotentialEnergyDissipationRate`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate))
minus the same contraction evaluated on the filtered state,
``\varepsilon_a^l`` ([`FilteredAvailablePotentialEnergyDissipationRate`](@ref)), with ``\bar q_i``
the closure's diffusive buoyancy flux low-pass filtered and ``\Upsilon^l = z^\star(\bar b) - z`` the
displacement potential of the filtered buoyancy. Filtering the flux, rather than recomputing it from
``\bar b``, is the same choice the [Filtered kinetic energy equation](@ref) makes for the viscous
flux; the [Filtered available potential energy equation](@ref) has the details.

The cross-scale APE flux ``\Pi_a``, which enters this budget as a source (``+\Pi_a``) and the
filtered one as a sink, is
[`AvailablePotentialEnergyCrossScaleFlux`](@ref Oceanostics.FilteredAvailablePotentialEnergyEquation.AvailablePotentialEnergyCrossScaleFlux),
defined in the [Filtered available potential energy equation](@ref) and re-exported here.

```@docs
Oceanostics.SubFilterAvailablePotentialEnergyEquation.SubFilterAvailablePotentialEnergyDissipationRate
```

## The subfilter conversion to kinetic energy

```math
\tau^l(w, b_r) = \overline{w b_r} - \bar w \, b_r^l ,
\qquad
b_r = b - b^\star(z) ,
\qquad
b_r^l = \bar b - b^\star(z)
```

is the rate at which the subfilter scales release their available potential energy to the subfilter
flow, computed by [`SubFilterAvailablePotentialToKineticEnergyConversion`](@ref). It is the subfilter
half of the split whose filtered half is
[`FilteredAvailablePotentialToKineticEnergyConversion`](@ref Oceanostics.FilteredAvailablePotentialEnergyEquation.FilteredAvailablePotentialToKineticEnergyConversion),
the two summing to ``\overline{w b_r}``. It enters this budget as ``-\tau^l(w, b_r)`` and the
[Sub-filter kinetic energy equation](@ref) as ``+\tau^l(w, b_r)``, so it is a reversible exchange rather
than a source or a sink. The reference profile is not filtered in either half, which is what
distinguishes it from a plain `subfilter_covariance` of ``w`` and ``b_r``.

```@docs
Oceanostics.SubFilterAvailablePotentialEnergyEquation.SubFilterAvailablePotentialToKineticEnergyConversion
```

The one term of the ``e_a^s`` budget still without a diagnostic is the reference-tendency correction
that appears when the reference profile evolves in time; with a fixed reference profile it vanishes
identically.
