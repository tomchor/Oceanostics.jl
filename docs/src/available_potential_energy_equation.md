# Available potential energy equation

The `AvailablePotentialEnergyEquation` module computes the share of the potential energy
``E_p = -bz`` of [the potential energy equation](potential_energy_equation.md) that the flow *can*
release. The other half, the reference state and the background potential energy ``E_b`` it carries,
lives in [the background potential energy equation](background_potential_energy_equation.md). The
reference height ``z^\star`` built there is the input to everything below, and this module re-exports
`reference_height`, `reference_buoyancy` and the reference-height methods so that either module can be
used on its own.

Oceanostics computes the available potential energy in its *local* form,
[Holliday & McIntyre (1981)](https://doi.org/10.1017/S0022112081001742):

```math
E_a(b, z) = \int_{z^\star}^{z} \left[b^\star(\tilde z) - b\right] \mathrm{d}\tilde z
          = \frac{g}{\rho_0}\int_{z^\star}^{z} \left[\rho - \rho^\star(\tilde z)\right] \mathrm{d}\tilde z .
```

In this form ``E_a`` is **non-negative everywhere in space** whenever the reference state is sorted from
the field itself, so it can be mapped as a field. Its volume integral recovers
``\int E_p - \int E_b`` in the continuum limit, although at finite ``\Delta z`` the two differ at second
order.

## Available potential energy

```@docs
Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy
```
