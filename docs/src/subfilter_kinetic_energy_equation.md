# Sub-filter kinetic energy equation

The `SubFilterKineticEnergyEquation` module provides diagnostics for the kinetic energy budget of the
*sub-filter* scales: the scales that a low-pass spatial filter ``\widetilde{(\,\cdot\,)}`` removes from
the flow. It is the companion of the [Filtered kinetic energy equation](@ref), which budgets the kinetic
energy ``K^l = \tfrac{1}{2}\,\tilde v_i\,\tilde v_i`` of the scales the filter *keeps*.

## The sub-filter kinetic energy budget

The sub-filter kinetic energy is half the trace of the sub-filter stress tensor
``\tau^r_{ij} = \widetilde{v_i v_j} - \tilde v_i \tilde v_j`` ([`subfilter_stress_tensor`](@ref)),

```math
K^s = \tfrac{1}{2}\,\tau^r_{ii}
    = \tfrac{1}{2}\left(\tau^r_{11} + \tau^r_{22} + \tau^r_{33}\right) ,
```

computed by [`SubFilterKineticEnergy`](@ref). Following the filtering framework of
[Aluie et al. (2018)](https://doi.org/10.1175/JPO-D-17-0100.1), its volume-integrated budget (with the
transport terms vanishing over a closed or periodic domain) reads

```math
\frac{d}{dt} \int K^s\, dV
    = \int \Pi_K\, dV
    + \int \tau(w, b_r)\, dV
    - \int \varepsilon^s\, dV ,
```

with two sources and one sink:

  - ``\Pi_K`` ([`KineticEnergyCrossScaleFlux`](@ref)) is the cross-scale kinetic-energy flux, the rate at
    which the filtered scales hand kinetic energy down to the sub-filter scales. It is the sink of the
    filtered-flow budget and the source of this one.
  - ``\tau(w, b_r) = \widetilde{w b_r} - \bar w\,b_r^l``
    ([`SubFilterAvailablePotentialToKineticEnergyConversion`](@ref Oceanostics.SubFilterAvailablePotentialEnergyEquation.SubFilterAvailablePotentialToKineticEnergyConversion))
    is the sub-filter buoyancy flux, which converts sub-filter available potential energy into
    sub-filter kinetic energy. It is defined in the
    [Sub-filter available potential energy equation](@ref), whose budget it is the sink of, and is
    re-exported here.
  - ``\varepsilon^s = \widetilde{\varepsilon} - \varepsilon^l`` is the sub-filter dissipation
    ([`SubFilterKineticEnergyDissipationRate`](@ref)): the filtered total dissipation
    ``\widetilde{\varepsilon}``
    ([`KineticEnergyDissipationRate`](@ref Oceanostics.KineticEnergyEquation.DissipationRate)) minus the
    dissipation ``\varepsilon^l`` of the filtered flow
    ([`FilteredKineticEnergyDissipationRate`](@ref)).

The [Rayleigh-Taylor instability](@ref rayleigh_taylor_example) example closes this budget for a
large-eddy simulation.

## Sub-filter kinetic energy

```@docs
Oceanostics.SubFilterKineticEnergyEquation.SubFilterKineticEnergy
```

## Sub-filter kinetic energy dissipation

```@docs
Oceanostics.SubFilterKineticEnergyEquation.SubFilterKineticEnergyDissipationRate
```
