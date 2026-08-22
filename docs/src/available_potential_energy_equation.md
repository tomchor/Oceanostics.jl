# Available potential energy equation

The `AvailablePotentialEnergyEquation` module computes the share of the potential energy
``e_p = -bz`` of [the potential energy equation](potential_energy_equation.md) that the flow *can*
release. The other half, the reference state and the background potential energy ``e_b`` it carries,
lives in [the background potential energy equation](background_potential_energy_equation.md). The
reference height ``z^\star`` built there is the input to everything below, and this module re-exports
`reference_height`, `reference_buoyancy` and the reference-height methods so that either module can be
used on its own.

Oceanostics computes the available potential energy in its *local* form,
[Holliday & McIntyre (1981)](https://doi.org/10.1017/S0022112081001742):

```math
e_a(b, z) = \int_{z^\star(b)}^{z} \left[b^\star(\tilde z,\, t) - b\right] \,\mathrm{d}\tilde z
          = \frac{g}{\rho_0}\int_{z^\star(\rho)}^{z} \left[\rho - \rho^\star(\tilde z,\, t)\right] \,\mathrm{d}\tilde z .
```

In this form ``e_a`` is **non-negative everywhere in space** whenever the reference state is sorted from
the field itself. Its volume integral recovers ``\int e_p - \int e_b`` in the continuum limit, although
at finite ``\Delta z`` the two differ at second order.

## Deriving the local available potential energy budget

The budget follows from the material derivative of ``e_a(b, z, t)`` along the flow
([Wenegrat, Chor & Barkan, 2026](https://arxiv.org/abs/2605.15879)):


```math
\frac{D e_a}{D t} = \left.\frac{\partial e_a}{\partial b}\right|_{z,t} \frac{D b}{D t}
                  + \left.\frac{\partial e_a}{\partial z}\right|_{b,t} \frac{D z}{D t}
                  + \left.\frac{\partial e_a}{\partial t}\right|_{z,b} \frac{D t}{D t},
```
which we can simplify to

```math
\frac{D e_a}{D t} = \left.\frac{\partial e_a}{\partial b}\right|_{z,t} \frac{D b}{D t}
                  + \left.\frac{\partial e_a}{\partial z}\right|_{b,t} w
                  + R ,
\qquad
R = \int_{z^\star}^{z} \partial_t b^\star(\tilde z, t) \, \mathrm{d}\tilde z .
```

Both partial derivatives come straight from the definition of ``e_a``. In the first, the boundary term
from moving the lower limit ``z^\star(b)`` drops out because ``b^\star(z^\star(b)) = b``, which leaves
the displacement potential ``\Upsilon``. The second is minus the buoyancy anomaly ``b_r`` the parcel
carries relative to the reference profile at its own height:

```math
\left.\frac{\partial e_a}{\partial b}\right|_{z} = z^\star - z = \Upsilon ,
\qquad
\left.\frac{\partial e_a}{\partial z}\right|_{b} = b^\star(z, t) - b = -b_r .
```

The buoyancy obeys [the tracer equation](tracer_equation.md), ``Db/Dt = -\partial_j q_j`` for an
unforced tracer with the closure's diffusive flux ``q_j``, so ``\Upsilon\,Db/Dt`` splits into a
transport divergence plus the contraction ``q_j\,\partial_j\Upsilon``. Writing the advective part as a
divergence as well (incompressibility), the local APE budget is

```math
\partial_t e_a = \underbrace{-\partial_j(u_j e_a)}_{\text{advection}}
                 \underbrace{-\,w b_r}_{\text{APE to KE conversion}}
                 \underbrace{-\,\partial_j(\Upsilon q_j)}_{\text{diffusive transport}}
                 \underbrace{-\,\varepsilon_a}_{\text{dissipation}}
                 + \underbrace{R}_{\text{reference tendency}} ,
\qquad
\varepsilon_a = -q_j \, \partial_j \Upsilon.
```

Similar to the two divergences, ``R`` redistribute ``e_a`` and vanish when integrated over a periodic or closed
domain ([Winters et al., 1995](https://doi.org/10.1017/S002211209500125X)). See the [Lock release](@ref lock_release_example)
example for an application of this budget.


## Terms and what is implemented

Five of the quantities above have diagnostics; the two transport terms and ``R`` have none.

| Quantity | Expression | Diagnostic |
|:---|:---|:---|
| Available potential energy | ``e_a = \int_{z^\star(b)}^{z} \left[b^\star(\tilde z,\, t) - b\right] \mathrm{d}\tilde z`` | [`AvailablePotentialEnergy`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy) |
| Displacement potential | ``\Upsilon = z^\star - z`` | [`DisplacementPotential`](@ref Oceanostics.AvailablePotentialEnergyEquation.DisplacementPotential) |
| Advection | ``\partial_j(u_j e_a)`` | not implemented |
| Buoyancy anomaly | ``b_r = b - b^\star(z,\, t)`` | [`ReferenceBuoyancyAnomaly`](@ref Oceanostics.AvailablePotentialEnergyEquation.ReferenceBuoyancyAnomaly) |
| APE to KE conversion | ``w b_r`` | [`AvailablePotentialToKineticEnergyConversion`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialToKineticEnergyConversion) |
| Diffusive transport | ``\partial_j(\Upsilon q_j)`` | not implemented |
| Dissipation | ``\varepsilon_a = -q_j \, \partial_j \Upsilon`` | [`AvailablePotentialEnergyDissipationRate`](@ref Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate) |
| Reference tendency | ``R = \int_{z^\star}^{z} \partial_t b^\star(\tilde z,\, t) \, \mathrm{d}\tilde z`` | not implemented |

The available potential energy converts to kinetic energy at a rate set by the buoyancy anomaly
``b_r``, not by the buoyancy itself. The remainder ``w \, b^\star(z,\, t)`` exchanges kinetic energy
with the background state. Their sum,

```math
w b = w b_r + w \, b^\star(z,\, t) ,
```

is [`PotentialToKineticEnergyConversion`](@ref Oceanostics.KineticEnergyEquation.PotentialEnergyConversion),
the conversion of the ``e_p`` budget. Both are formed from the same face-centered product, so the split
holds cell by cell; under a volume integral the distinction goes away entirely, since
``w \, b^\star(z,\, t)`` is a divergence.

Every diagnostic here is built on the reference state of
[the background potential energy equation](background_potential_energy_equation.md), whose
[`reference_height`](@ref Oceanostics.BackgroundPotentialEnergyEquation.reference_height) supplies
``z^\star``,
[`reference_buoyancy`](@ref Oceanostics.BackgroundPotentialEnergyEquation.reference_buoyancy) the
profile ``b^\star`` paired with it, and
[`reference_buoyancy_at_height`](@ref Oceanostics.BackgroundPotentialEnergyEquation.reference_buoyancy_at_height)
the ``b^\star(z,\, t)`` that ``b_r`` is measured against; all three are re-exported here, as is
[`DiffusiveVerticalBuoyancyFlux`](@ref Oceanostics.PotentialEnergyEquation.DiffusiveVerticalBuoyancyFlux),
the flux ``\Phi = -q_3`` that ``\varepsilon_a`` leaves out of the diapycnal mixing rate of
[Winters et al. (1995)](https://doi.org/10.1017/S002211209500125X).

## Summary of ``e_a`` equation terms


```@docs
Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergy
Oceanostics.AvailablePotentialEnergyEquation.DisplacementPotential
Oceanostics.AvailablePotentialEnergyEquation.ReferenceBuoyancyAnomaly
Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialToKineticEnergyConversion
Oceanostics.AvailablePotentialEnergyEquation.AvailablePotentialEnergyDissipationRate
```
