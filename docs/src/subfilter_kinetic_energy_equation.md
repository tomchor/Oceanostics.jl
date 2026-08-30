# Subfilter kinetic energy equation

The `SubFilterKineticEnergyEquation` module provides diagnostics for the kinetic energy budget of the
*subfilter* scales: the scales that a low-pass spatial filter ``\overline{(\,\cdot\,)}`` removes from
the flow. It is the companion of the [Filtered kinetic energy equation](@ref), which budgets the kinetic
energy ``e_k^l = \tfrac{1}{2}\,\bar u_i\,\bar u_i`` of the scales the filter *keeps*.

## The subfilter kinetic energy budget

The subfilter kinetic energy is half the trace of the subfilter stress tensor
``\tau^r_{ij} = \overline{u_i u_j} - \bar u_i \bar u_j`` ([`subfilter_stress_tensor`](@ref)),

```math
e_k^s = \tfrac{1}{2}\,\tau^r_{ii}
    = \tfrac{1}{2}\left(\tau^r_{11} + \tau^r_{22} + \tau^r_{33}\right) ,
```

computed by [`SubFilterKineticEnergy`](@ref). Following the filtering framework of
[Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1), its volume-integrated budget (with the
transport terms vanishing over a closed or periodic domain) reads

```math
\frac{d}{dt} \int e_k^s\, \mathrm{d}V
    = \int \Pi_k\, \mathrm{d}V
    + \int \tau^l(w, b_r)\, \mathrm{d}V
    - \int \varepsilon_k^s\, \mathrm{d}V ,
```

with two sources and one sink:

  - ``\Pi_k`` ([`KineticEnergyCrossScaleFlux`](@ref)) is the cross-scale kinetic-energy flux, the rate at
    which the filtered scales hand kinetic energy down to the subfilter scales. It is the sink of the
    filtered-flow budget and the source of this one.
  - ``\tau^l(w, b_r) = \overline{w b_r} - \bar w\,b_r^l``
    ([`SubFilterAvailablePotentialToKineticEnergyConversion`](@ref Oceanostics.SubFilterAvailablePotentialEnergyEquation.SubFilterAvailablePotentialToKineticEnergyConversion))
    is the subfilter buoyancy flux, which converts subfilter available potential energy into
    subfilter kinetic energy. It is defined in the
    [Subfilter available potential energy equation](@ref), whose budget it is the sink of, and is
    re-exported here.
  - ``\varepsilon_k^s = \overline{\varepsilon_k} - \varepsilon_k^l`` is the subfilter dissipation
    ([`SubFilterKineticEnergyDissipationRate`](@ref)): the filtered total dissipation
    ``\overline{\varepsilon_k}``
    ([`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)) minus the
    dissipation ``\varepsilon_k^l`` of the filtered flow
    ([`FilteredKineticEnergyDissipationRate`](@ref)).

The [Rayleigh-Taylor instability](@ref rayleigh_taylor_example) example closes this budget for a
large-eddy simulation.

## Subfilter kinetic energy

```@docs
Oceanostics.SubFilterKineticEnergyEquation.SubFilterKineticEnergy
```

## Subfilter kinetic energy dissipation

```@docs
Oceanostics.SubFilterKineticEnergyEquation.SubFilterKineticEnergyDissipationRate
```
