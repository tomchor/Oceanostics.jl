# Tracer equation

The `TracerEquation` module provides diagnostics corresponding to individual terms
in the tracer conservation equation:

```math
\partial_t c = -\partial_j (u_j c) + \partial_j q^c_j + F^c
```

where ``c`` is a tracer, ``u_j`` is the velocity, ``q^c_j`` is the diffusive tracer flux,
and ``F^c`` is a forcing term.

Each diagnostic wraps an Oceananigans kernel and returns a `KernelFunctionOperation`
at `(Center, Center, Center)`.

## Advection

```@docs
Oceanostics.TracerEquation.Advection
```

## Diffusion

```@docs
Oceanostics.TracerEquation.Diffusion
Oceanostics.TracerEquation.ImmersedDiffusion
Oceanostics.TracerEquation.TotalDiffusion
```

## Forcing

```@docs
Oceanostics.TracerEquation.Forcing
```
