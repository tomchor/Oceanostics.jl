# Filtered available potential energy equation

The `FilteredAvailablePotentialEnergyEquation` module provides diagnostics for the available potential
energy of the *filtered* buoyancy field. It is the potential-energy counterpart of the
[Filtered kinetic energy equation](@ref). Following the filtered APE framework of
[Wenegrat, Chor & Barkan (2026)](https://arxiv.org/abs/2605.15879), we define the APE
of the filtered flow as

```math
e_a^l = e_a(\bar b, z, t) = \int_{z^\star(\bar b, t)}^{z} \left[b^\star(\tilde z, t) - \bar b\right] \mathrm{d}\tilde z ,
```
computed by [`FilteredAvailablePotentialEnergy`](@ref). Note that the reference profile ``b^\star(z^\star)``
is the reference state of the *full buoyancy* ``b``, rather than sorted from ``\bar b``.
Because of this, it is necessary to look up the position of a buoyancy parcel in the filtered fields
in the full reference profile ``b^\star(z^\star)`` in calculating some terms, which makes it necessary
to use [`ProfileLookup`](@ref Oceanostics.BackgroundPotentialEnergyEquation.ProfileLookup), since it's
the only method of obtaining the reference height that has this capability.


## Deriving the filtered available potential energy equation

The budget of ``e_a^l`` retraces the
[full-APE derivation](@ref "Deriving the local available potential energy equation") with the
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

We can then evaluate the partial derivatives above from the definition of ``e_a^l``, similarly to the
``e_a`` derivation. In the first we get an "empty" integral from ``z^\star`` to ``z``, which produces
the displacement potential of the filtered buoyancy ``\Upsilon^l``. The second simplifies to the integrand of ``e_a^l``, producing
the buoyancy anomaly ``b_r^l``:

```math
\left.\frac{\partial e_a^l}{\partial \bar b}\right|_{z} = z^\star(\bar b, t) - z = \Upsilon^l ,
\qquad
\left.\frac{\partial e_a^l}{\partial z}\right|_{\bar b} = b^\star(z, t) - \bar b = -b_r^l .
```

Note that ``\Upsilon^l`` is different from ``\Upsilon`` since it includes ``z^\star(\bar b, t)`` and *not*
``z^\star(b, t)``. The former is the reference height of the filtered buoyancy: the height at which a parcel
of the _filtered_ buoyancy would sit in the reference profile _unfiltered_ buoyancy.
Similarly, the buoyancy anomaly ``b_r^l`` is measured against the reference profile of the full, unfiltered
buoyancy ``b^\star(z, t)`` -- the reference profile of the _filtered_ buoyancy does not appear anywhere in this equation.

The material derivative of the filtered buoyancy comes from filtering
[the tracer equation](tracer_equation.md) of ``b`` (the filter commutes with derivatives) and
splitting the filtered advective flux into a resolved and a subfilter part,
``\overline{u_i b} = \bar u_i \, \bar b + \tau(u_i, b)``:

```math
\frac{D^l \bar b}{D t} = -\partial_i \tau(u_i, b) - \partial_i \bar q_i ,
\qquad
\tau(u_i, b) = \overline{b u_i} - \bar b \, \bar u_i ,
```

where ``\tau(u_i, b)`` is the subfilter buoyancy flux and ``\bar q_i`` is the filtered closure's diffusive flux
low-pass filtered. The term ``\Upsilon^l \, D^l \bar b / D t`` splits into two transport divergences
plus the contractions ``\tau(u_i, b) \, \partial_i \Upsilon^l`` and ``\bar q_i \, \partial_i \Upsilon^l``.
Writing the advective part as a divergence as well (``\partial_i \bar u_i = 0``, since filtering
preserves incompressibility), the filtered APE budget is

```math
\partial_t e_a^l = \underbrace{-\partial_i(\bar u_i e_a^l)}_{\text{advection}}
                   \underbrace{-\,\bar w \, b_r^l}_{\text{APE to KE conversion}}
                   \underbrace{-\,\Pi_a}_{\text{cross-scale flux}}
                   \underbrace{-\,\partial_i\!\left[\Upsilon^l \left(\tau(u_i, b) + \bar q_i\right)\right]}_{\text{subfilter and diffusive transport}}
                   \underbrace{-\,\varepsilon_a^l}_{\text{dissipation}}
                   + \underbrace{R^l}_{\text{reference tendency}} ,
```

with

```math
\Pi_a = -\tau(u_i, b) \, \partial_i \Upsilon^l ,
\qquad
\varepsilon_a^l = -\bar q_i \, \partial_i \Upsilon^l .
```

Term by term this is the full-field budget evaluated on the filtered state plus extra terms
related to the filter (``\Pi_a`` and ``-\partial_j\Upsilon^l \tau(u_i, b)``). ``\Pi_a`` reappears
with the opposite sign in the [Subfilter available potential energy equation](@ref), which makes
it a transfer across the filter scale rather than a source or a sink, and the conversion
``\bar w \, b_r^l`` likewise reappears with the opposite sign in the
[filtered kinetic energy](@ref "Filtered kinetic energy equation") budget.

Importantly, while ``R`` integrates to zero in a closed domain ([Winters et al., 1995](https://doi.org/10.1017/S002211209500125X)),
``R^l`` does not inherit that property. Given its subfilter counterpart ``R^s = \overline{R} - R^l``
appears in the [Subfilter available potential energy equation](@ref), we get that
``\int R^l \, \mathrm{d}V = -\int R^s \, \mathrm{d}V``. Thus, interestingly, an evolving reference profile
redistributes APE across the filter scale as well as in space
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)). With a reference profile held
fixed in time (implemented here with a [`ProfileLookup`](@ref Oceanostics.BackgroundPotentialEnergyEquation.ProfileLookup)
holding plain arrays) ``R^l`` vanishes identically.

## Terms and diagnostics

Five of the quantities above have diagnostics; the two transport divergences, the anomaly
``b_r^l``, and the reference tendency have none.

| Quantity | Expression | Diagnostic |
|:---|:---|:---|
| Filtered available potential energy | ``e_a^l = e_a(\bar b, z)`` | [`FilteredAvailablePotentialEnergy`](@ref) |
| Displacement potential | ``\Upsilon^l = z^\star(\bar b) - z`` | [`FilteredAvailablePotentialEnergyDisplacementPotential`](@ref) |
| Advection | ``\partial_i(\bar u_i e_a^l)`` | not implemented |
| Buoyancy anomaly | ``b_r^l = \bar b - b^\star(z, t)`` | not implemented |
| APE to KE conversion | ``\bar w \, b_r^l`` | [`FilteredAvailablePotentialToKineticEnergyConversion`](@ref) |
| Cross-scale flux | ``\Pi_a = -\tau(u_i, b) \, \partial_i \Upsilon^l`` | [`AvailablePotentialEnergyCrossScaleFlux`](@ref) |
| Subfilter and diffusive transport | ``\partial_i\left[\Upsilon^l (\tau(u_i, b) + \bar q_i)\right]`` | not implemented |
| Dissipation | ``\varepsilon_a^l = -\bar q_i \, \partial_i \Upsilon^l`` | [`FilteredAvailablePotentialEnergyDissipationRate`](@ref) |
| Reference tendency | ``R^l = \int_{z^\star(\bar b)}^{z} \partial_t b^\star(\tilde z, t) \, \mathrm{d}\tilde z`` | not implemented |

``\Upsilon^l`` also answers to `DisplacementPotential`, and ``\varepsilon_a^l`` to `DissipationRate`.
Both aliases are scoped to this module, as their full-field namesakes are to
[the available potential energy equation](available_potential_energy_equation.md): `using
Oceanostics.FilteredAvailablePotentialEnergyEquation` brings them in, `using Oceanostics` does not,
since unprefixed neither name says which budget it belongs to nor at which scale.

## Summary of ``e_a^l`` equation terms

```@docs
Oceanostics.FilteredAvailablePotentialEnergyEquation.FilteredAvailablePotentialEnergy
Oceanostics.FilteredAvailablePotentialEnergyEquation.FilteredAvailablePotentialEnergyDisplacementPotential
Oceanostics.FilteredAvailablePotentialEnergyEquation.FilteredAvailablePotentialToKineticEnergyConversion
Oceanostics.FilteredAvailablePotentialEnergyEquation.AvailablePotentialEnergyCrossScaleFlux
Oceanostics.FilteredAvailablePotentialEnergyEquation.FilteredAvailablePotentialEnergyDissipationRate
```
