# Filtered available potential energy equation

The `FilteredAvailablePotentialEnergyEquation` module provides diagnostics for the available potential
energy of the *filtered* buoyancy field: the share of the local APE ``e_a`` of
[the available potential energy equation](available_potential_energy_equation.md) that the scales a
low-pass spatial filter ``\overline{(\,\cdot\,)}`` keeps would carry on their own, following the
filtered APE framework of [Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879). It is
the potential-energy counterpart of the [Filtered kinetic energy equation](@ref), and the
[Sub-filter available potential energy equation](@ref) budgets what the filter removes.

## The available potential energy of the filtered buoyancy

With ``\bar b`` the filtered buoyancy, the APE of the filtered field is the local APE functional
evaluated on ``\bar b`` rather than on ``b``,

```math
e_a^l = e_a(\bar b, z) = \int_{z^\star(\bar b)}^{z} \left[b^\star(\tilde z) - \bar b\right] \mathrm{d}\tilde z ,
```

computed by [`FilteredAvailablePotentialEnergy`](@ref). The reference profile ``(b^\star, z^\star)`` it
is measured against is **shared with the full field**, ordinarily the sorted state of the full buoyancy
``b``, rather than sorted from ``\bar b`` itself: only then are ``e_a(\bar b, z)`` and
``e_a(b, z)`` comparable, and their difference the sub-filter APE. Looking a field up in a profile it
did not produce is exactly what
[`ProfileLookup`](@ref Oceanostics.BackgroundPotentialEnergyEquation.ProfileLookup) was built for, so
these diagnostics accept only that reference-height method: the default sorts the model's own buoyancy
into a [`VerticalSort`](@ref Oceanostics.BackgroundPotentialEnergyEquation.VerticalSort) column on
every `compute!`, a column you built yourself can be shared across diagnostics, and a profile given as
plain arrays holds the reference state fixed in time (which also makes the diagnostics sort-free).

```@docs
Oceanostics.FilteredAvailablePotentialEnergyEquation.FilteredAvailablePotentialEnergy
```

## Deriving the filtered available potential energy equation

The budget of ``e_a^l`` retraces the
[full-field derivation](@ref "Deriving the local available potential energy equation") with the
filtered buoyancy taking the place of ``b`` and the material derivative taken along the *filtered*
flow, ``D^l/Dt = \partial_t + \bar u_i \partial_i``
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)):

```math
\frac{D^l e_a^l}{D t} = \left.\frac{\partial e_a^l}{\partial \bar b}\right|_{z,t} \frac{D^l \bar b}{D t}
                      + \left.\frac{\partial e_a^l}{\partial z}\right|_{\bar b,t} \frac{D^l z}{D t}
                      + \left.\frac{\partial e_a^l}{\partial t}\right|_{z,\bar b} \frac{D^l t}{D t} ,
```

which we can simplify to

```math
\frac{D^l e_a^l}{D t} = \left.\frac{\partial e_a^l}{\partial \bar b}\right|_{z,t} \frac{D^l \bar b}{D t}
                      + \left.\frac{\partial e_a^l}{\partial z}\right|_{\bar b,t} \bar w
                      + R^l ,
\qquad
R^l = \int_{z^\star(\bar b)}^{z} \partial_t b^\star(\tilde z, t) \, \mathrm{d}\tilde z .
```

Since ``e_a^l`` is the function ``e_a`` evaluated at ``(\bar b, z)``, both partial derivatives are
the full-field ones taken at the filtered state; in the first, the boundary term from moving the
lower limit ``z^\star(\bar b)`` again drops out, now because ``b^\star(z^\star(\bar b)) = \bar b``.
They give the displacement potential and the buoyancy anomaly *of the filtered buoyancy*:

```math
\left.\frac{\partial e_a^l}{\partial \bar b}\right|_{z} = z^\star(\bar b) - z = \Upsilon^l ,
\qquad
\left.\frac{\partial e_a^l}{\partial z}\right|_{\bar b} = b^\star(z, t) - \bar b = -b_r^l .
```

Differentiating in ``z`` at fixed ``\bar b`` never touches the reference profile, so the anomaly
that appears is ``b_r^l = \bar b - b^\star(z, t)``, the filtered buoyancy against the *unfiltered*
profile, and not ``\overline{b_r}``; the note in
[Conversion to filtered kinetic energy](@ref) below returns to this distinction.

The material derivative of the filtered buoyancy comes from filtering
[the tracer equation](tracer_equation.md) of ``b`` (the filter commutes with derivatives) and
splitting the filtered advective flux into a resolved and a sub-filter part,
``\overline{u_i b} = \bar u_i \, \bar b + \tau_i``:

```math
\frac{D^l \bar b}{D t} = -\partial_i \tau_i - \partial_i \bar q_i ,
\qquad
\tau_i = \overline{b u_i} - \bar b \, \bar u_i ,
```

where ``\tau_i`` is the sub-filter buoyancy flux and ``\bar q_i`` the closure's diffusive flux
low-pass filtered, not recomputed from ``\bar b`` (the
[dissipation section](@ref "The available potential energy dissipation of the filtered buoyancy")
draws that distinction), so ``\Upsilon^l \, D^l \bar b / D t`` splits into two transport divergences
plus the contractions ``\tau_i \, \partial_i \Upsilon^l`` and ``\bar q_i \, \partial_i \Upsilon^l``.
Writing the advective part as a divergence as well (``\partial_i \bar u_i = 0``, since filtering
preserves incompressibility), the filtered APE budget is

```math
\partial_t e_a^l = \underbrace{-\partial_i(\bar u_i e_a^l)}_{\text{advection}}
                   \underbrace{-\,\bar w \, b_r^l}_{\text{APE to KE conversion}}
                   \underbrace{-\,\Pi_a}_{\text{cross-scale flux}}
                   \underbrace{-\,\partial_i\!\left[\Upsilon^l \left(\tau_i + \bar q_i\right)\right]}_{\text{sub-filter and diffusive transport}}
                   \underbrace{-\,\varepsilon_a^l}_{\text{dissipation}}
                   + \underbrace{R^l}_{\text{reference tendency}} ,
```

with

```math
\Pi_a = -\tau_i \, \partial_i \Upsilon^l ,
\qquad
\varepsilon_a^l = -\bar q_i \, \partial_i \Upsilon^l .
```

Term by term this is the full-field budget evaluated on the filtered state, plus one genuinely new
term: the cross-scale flux ``\Pi_a``, which vanishes together with ``\tau_i`` as the filter tends to
the identity while every other term collapses onto its full-field counterpart. ``\Pi_a`` reappears
with the opposite sign in the [Sub-filter available potential energy equation](@ref), which is what
makes it a transfer across the filter scale rather than a source or a sink. The conversion
``\bar w \, b_r^l`` likewise reappears with the opposite sign in the
[filtered kinetic energy](@ref "Filtered kinetic energy equation") budget, where it and
``\bar w \, b^\star(z)``, the exchange with the background state, make up the buoyancy production
``\bar w \bar b = \bar w \, b_r^l + \bar w \, b^\star``, mirroring the split
``w b = w b_r + w \, b^\star`` of the full-field budget.

The advective and transport divergences again redistribute ``e_a^l`` and integrate to zero over a
periodic or closed domain. The reference tendency does not inherit that property:
``\int R \, \mathrm{d}V = 0`` ([Winters et al., 1995](https://doi.org/10.1017/S002211209500125X))
constrains only the sum of ``R^l`` and its sub-filter counterpart ``R^s = \overline{R} - R^l``, that
is, ``\int R^l \, \mathrm{d}V = -\int R^s \, \mathrm{d}V``, so an evolving reference profile
redistributes APE across the filter scale as well as in space
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)). With a reference profile held
fixed in time (a [`ProfileLookup`](@ref Oceanostics.BackgroundPotentialEnergyEquation.ProfileLookup)
holding plain arrays) ``R^l`` vanishes identically.

## Terms and diagnostics

Five of the quantities above have diagnostics; the two transport divergences, the anomaly
``b_r^l``, and the reference tendency have none.

| Quantity | Expression | Diagnostic |
|:---|:---|:---|
| Filtered available potential energy | ``e_a^l = e_a(\bar b, z)`` | [`FilteredAvailablePotentialEnergy`](@ref) |
| Displacement potential | ``\Upsilon^l = z^\star(\bar b) - z`` | [`AvailablePotentialEnergyDisplacementPotential`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDisplacementPotential) on a reference height built from ``\bar b`` |
| Advection | ``\partial_i(\bar u_i e_a^l)`` | not implemented |
| Buoyancy anomaly | ``b_r^l = \bar b - b^\star(z, t)`` | not implemented |
| APE to KE conversion | ``\bar w \, b_r^l`` | [`FilteredAvailablePotentialToKineticEnergyConversion`](@ref) |
| Cross-scale flux | ``\Pi_a = -\tau_i \, \partial_i \Upsilon^l`` | [`AvailablePotentialEnergyCrossScaleFlux`](@ref) |
| Sub-filter and diffusive transport | ``\partial_i\left[\Upsilon^l (\tau_i + \bar q_i)\right]`` | not implemented |
| Dissipation | ``\varepsilon_a^l = -\bar q_i \, \partial_i \Upsilon^l`` | [`FilteredAvailablePotentialEnergyDissipationRate`](@ref) |
| Reference tendency | ``R^l = \int_{z^\star(\bar b)}^{z} \partial_t b^\star(\tilde z, t) \, \mathrm{d}\tilde z`` | not implemented |

## The available potential energy dissipation of the filtered buoyancy

The diffusive sink of the ``e_a^l`` budget is

```math
\varepsilon_a^l = -\bar q_i \, \partial_i \Upsilon^l ,
\qquad
\Upsilon^l = z^\star(\bar b) - z ,
```

computed by [`FilteredAvailablePotentialEnergyDissipationRate`](@ref): the full-field contraction
``\varepsilon_a = -q_i \partial_i \Upsilon``
([`AvailablePotentialEnergyDissipationRate`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate))
evaluated on the filtered state, with ``\bar q_i`` the closure's diffusive buoyancy flux low-pass
filtered and ``\Upsilon^l`` the displacement potential
([`DisplacementPotential`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDisplacementPotential))
of the filtered buoyancy. The flux is filtered, not recomputed from ``\bar b``: the filtered buoyancy
equation carries the divergence of the filtered flux, so ``-\bar q_i \partial_i \Upsilon^l`` is the
dissipation that appears in the filtered-state budget. The two forms agree for a constant diffusivity
and differ once ``\kappa`` varies in space, the same distinction the
[Filtered kinetic energy equation](@ref) draws for the viscous flux.

```@docs
Oceanostics.FilteredAvailablePotentialEnergyEquation.FilteredAvailablePotentialEnergyDissipationRate
```

## Cross-scale available potential energy flux

```math
\Pi_a = -\tau_i \, \partial_i \Upsilon^l , \qquad
\tau_i = \overline{b u_i} - \bar b \, \bar u_i , \qquad
\Upsilon^l = z^\star(\bar b) - z
```

is the rate at which the filter transfers available potential energy from the filtered to the
sub-filter scales, the APE analogue of the
[cross-scale kinetic energy flux](filtered_kinetic_energy_equation.md) ``\Pi_k = -\tau^{ij}\bar S^{ij}``:
the sub-filter buoyancy flux takes the place of the sub-filter stress, and ``\nabla\Upsilon^l`` takes
the place of the resolved strain. ``\Pi_a > 0`` is forward (downscale) transfer. It enters the filtered
APE budget as ``-\Pi_a`` and the sub-filter one as ``+\Pi_a``, which is what makes it a transfer rather
than a source or a sink ([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)).

```@docs
Oceanostics.FilteredAvailablePotentialEnergyEquation.AvailablePotentialEnergyCrossScaleFlux
```

## Conversion to filtered kinetic energy

The filtered APE and the [filtered kinetic energy](@ref "Filtered kinetic energy equation") exchange
energy at a rate

```math
\bar w \, b_r^l ,
\qquad
b_r^l = \bar b - b^\star(z) ,
```

computed by [`FilteredAvailablePotentialToKineticEnergyConversion`](@ref). Here ``b_r^l`` is the
buoyancy anomaly of the filtered field against the reference state, with ``b^\star(z)`` the reference
profile read at the parcel's **own height** ``z`` rather than at the reference height ``z^\star`` its
buoyancy would take it to — the inverse of the map every other diagnostic on this page uses.

The term appears in the filtered APE budget as ``-\bar w\,b_r^l`` and in the filtered kinetic energy
budget as ``+\bar w\,b_r^l`` ([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879),
Eqs. 2.10 and 3.2), so it is a reversible exchange rather than a source or a sink: ``\bar w\,b_r^l > 0``
converts filtered APE into filtered KE.

!!! note "The reference profile is not filtered"
    ``b_r^l = \bar b - b^\star(z)`` pairs the *filtered* buoyancy with the *unfiltered* reference
    profile, consistent with ``e_a^l`` itself being measured against the full field's reference state.
    It is not ``\overline{b_r} = \bar b - \overline{b^\star(z)}``, which filters the reference too.
    The two differ once the filter acts in the vertical; for a purely horizontal filter they coincide,
    since ``b^\star`` is a function of ``z`` alone.

```@docs
Oceanostics.FilteredAvailablePotentialEnergyEquation.FilteredAvailablePotentialToKineticEnergyConversion
```
