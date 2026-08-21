# Filtered available potential energy equation

The `FilteredAvailablePotentialEnergyEquation` module provides diagnostics for the available potential
energy of the *filtered* buoyancy field: the share of the local APE ``e_a`` of
[the available potential energy equation](available_potential_energy_equation.md) that the scales a
low-pass spatial filter ``\widetilde{(\,\cdot\,)}`` keeps would carry on their own, following the
filtered APE framework of [Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879). It is
the potential-energy counterpart of the [Filtered kinetic energy equation](@ref), and the
[Sub-filter available potential energy equation](@ref) budgets what the filter removes.

## The available potential energy of the filtered buoyancy

With ``\tilde b`` the filtered buoyancy, the APE of the filtered field is the local APE functional
evaluated on ``\tilde b`` rather than on ``b``,

```math
e_a^l = e_a(\tilde b, z) = \int_{z^\star(\tilde b)}^{z} \left[b^\star(\tilde z) - \tilde b\right] \mathrm{d}\tilde z ,
```

computed by [`FilteredAvailablePotentialEnergy`](@ref). The reference profile ``(b^\star, z^\star)`` it
is measured against is **shared with the full field**, ordinarily the sorted state of the full buoyancy
``b``, rather than sorted from ``\tilde b`` itself: only then are ``e_a(\tilde b, z)`` and
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

## The available potential energy dissipation of the filtered buoyancy

The diffusive sink of the ``e_a^l`` budget is

```math
\varepsilon_a^l = -\tilde q_i \, \partial_i \Upsilon^l ,
\qquad
\Upsilon^l = z^\star(\tilde b) - z ,
```

computed by [`FilteredAvailablePotentialEnergyDissipationRate`](@ref): the full-field contraction
``\varepsilon_a = -q_i \partial_i \Upsilon``
([`AvailablePotentialEnergyDissipationRate`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate))
evaluated on the filtered state, with ``\tilde q_i`` the closure's diffusive buoyancy flux low-pass
filtered and ``\Upsilon^l`` the displacement potential
([`BuoyancyDisplacementPotential`](@ref Oceanostics.AvailablePotentialEnergyEquation.BuoyancyDisplacementPotential))
of the filtered buoyancy. The flux is filtered, not recomputed from ``\tilde b``: the filtered buoyancy
equation carries the divergence of the filtered flux, so ``-\tilde q_i \partial_i \Upsilon^l`` is the
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
energy at a rate ``\bar w\,b_r^l``, computed by
[`FilteredAvailablePotentialToKineticEnergyConversion`](@ref). It follows from differentiating
``e_a^l`` along the filtered flow. Writing the local APE of an arbitrary buoyancy ``b`` at a height
``z`` as in [the available potential energy equation](available_potential_energy_equation.md),

```math
e_a(b, z) = \int_{z^\star(b)}^{z} \left[b^\star(\tilde z) - b\right] \mathrm{d}\tilde z ,
```

it has two partial derivatives. Differentiating with respect to the parcel's buoyancy moves the lower
limit, but the integrand vanishes there, because a parcel carries ``b = b^\star(z^\star(b))`` by
construction; what is left is the displacement potential,

```math
\left.\frac{\partial e_a}{\partial b}\right|_z = z^\star(b) - z = \Upsilon .
```

Differentiating with respect to the parcel's height moves the upper limit instead, and there the
integrand does not vanish:

```math
\left.\frac{\partial e_a}{\partial z}\right|_b = b^\star(z) - b = -b_r ,
\qquad
b_r = b - b^\star(z) ,
```

which defines the buoyancy anomaly ``b_r`` relative to the resorted reference state
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879), Eq. 2.7). Note the two derivatives
sample the reference profile differently: ``\Upsilon`` needs the reference *height* of the parcel's own
buoyancy, ``z^\star(b)``, while ``b_r`` needs the reference *buoyancy* at the parcel's own height,
``b^\star(z)`` — the inverse of the same monotone map.

Applying the chain rule to ``e_a^l = e_a(\bar b, z)`` along the filtered flow, with
``D_l/Dt = \partial_t + \bar u_i \partial_i`` and hence ``D_l z / Dt = \bar w``, gives

```math
\frac{D_l e_a^l}{Dt}
    = \left.\frac{\partial e_a}{\partial b}\right|_{\bar b} \frac{D_l \bar b}{Dt}
    + \left.\frac{\partial e_a}{\partial z}\right|_{\bar b} \bar w
    + R^l
    = \Upsilon^l \frac{D_l \bar b}{Dt}
    - \underbrace{\bar w\, b_r^l}_{\text{conversion}}
    + R^l ,
```

where ``R^l`` collects the change of ``e_a^l`` due to the reference profile itself evolving in time
(Eq. 2.4; no diagnostic yet). Expanding ``D_l\bar b/Dt`` with the filtered buoyancy equation turns the
first term into the cross-scale flux, the transport, and the dissipation, recovering the budget
(Eq. 2.10)

```math
\frac{D_l e_a^l}{Dt} = -\bar w\,b_r^l - \Pi_a - \partial_i F_i^l + R^l - \varepsilon_a^l .
```

The conversion enters the filtered kinetic energy budget with the opposite sign (Eq. 3.2), so it is a
reversible exchange rather than a source or a sink: ``\bar w\,b_r^l > 0`` converts filtered APE into
filtered KE.

!!! note "The reference profile is not filtered"
    ``b_r^l = \bar b - b^\star(z)`` pairs the *filtered* buoyancy with the *unfiltered* reference
    profile, which is what the derivative above produces, and it is consistent with ``e_a^l`` itself
    being measured against the full field's reference state. It is not ``\overline{b_r} = \bar b -
    \overline{b^\star(z)}``, which filters the reference too. The two differ once the filter acts in
    the vertical; for a purely horizontal filter they coincide, since ``b^\star`` is a function of
    ``z`` alone.

```@docs
Oceanostics.FilteredAvailablePotentialEnergyEquation.FilteredAvailablePotentialToKineticEnergyConversion
```
