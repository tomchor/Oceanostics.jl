# Tracer variance equation

The `TracerVarianceEquation` module provides diagnostics for the tracer variance
budget. For a tracer ``c``, the variance ``c^2`` evolves as:

```math
\partial_t c^2 = 2c\,\partial_t c = 2c\,(-\partial_j(u_j c) + \partial_j F_j + F^c)
```

The module decomposes this into tendency, diffusion, and dissipation rate terms.

## Tendency

```@docs
Oceanostics.TracerVarianceEquation.TracerVarianceTendency
```

## Diffusion

```@docs
Oceanostics.TracerVarianceEquation.TracerVarianceDiffusion
```

## Dissipation rate

```@docs
Oceanostics.TracerVarianceEquation.TracerVarianceDissipationRate
```
