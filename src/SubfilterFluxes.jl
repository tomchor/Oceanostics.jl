module SubfilterFluxes
using DocStringExtensions

export SubfilterFlux

using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: Field
using Oceananigans.AbstractOperations: @at

"""
    $(SIGNATURES)

Return a lazy `AbstractOperation` for the **generalized subfilter covariance** (second moment) of
two fields `a` and `b` under a low-pass spatial `filter` (overbar),

```math
\\tau(a, b) = \\overline{a\\,b} - \\bar{a}\\,\\bar{b},
```

co-located at `loc`. Here `filter(ψ) ≡ ψ̄` is a normalized local average (e.g. an Oceanostics
[`GaussianFilter`](@ref) or [`BoxFilter`](@ref)) that splits a field into a resolved part `ψ̄` and a
subfilter fluctuation `ψ′ = ψ - ψ̄`. The quantity `τ(a, b)` is the part of the product `ab` that the
filtered fields `ā b̄` cannot represent on their own — i.e. the transport/stress carried by scales
smaller than the filter width (Aluie et al. 2018, *J. Phys. Oceanogr.*,
doi:10.1175/JPO-D-17-0100.1).

Two common special cases are:

  - **Subfilter tracer flux** — with `a = uᵢ` (a velocity component) and `b = c` (a tracer),
    `τ(uᵢ, c) = overline(uᵢ c) - ūᵢ c̄` is the flux of `c` carried by unresolved scales.
  - **Subfilter momentum stress** — with `a = uᵢ` and `b = uⱼ`,
    `τ(uᵢ, uⱼ) = overline(uᵢ uⱼ) - ūᵢ ūⱼ` is the subfilter (subgrid-scale) Reynolds-type stress
    tensor component.

# Arguments

  - `a`, `b`: the two operands (`Field`s or `AbstractOperation`s). They are interpolated to the
    common location `loc` before being multiplied and filtered, so they may originally live at
    different staggered-grid locations (e.g. a `Face`-located velocity and a `Center`-located
    tracer).
  - `filter`: a **function** that maps a field to its filtered counterpart, i.e. `ψ -> ψ̄`. Build it
    as a closure over an Oceanostics filter, fixing every keyword except the field, e.g.
    `filter = ψ -> GaussianFilter(ψ; dims=(1, 2), σ=0.1)`.

# Keyword arguments

  - `loc = (Center, Center, Center)`: the location triple (types, not instances) at which the two
    operands are co-located and the covariance is returned.

The filtered pieces (`overline(a b)`, `ā`, `b̄`) are materialized as `Field`s so the separable
filter's fast staged path fires; the returned object is a lazy `AbstractOperation` over those
computed fields, ready to be wrapped in `Field(...)`, reduced with `Integral`, or handed to an
`OutputWriter`.
"""
function SubfilterFlux(a, b, filter; loc = (Center, Center, Center))
    a_loc = Field(@at loc a)                                   # co-locate operands at `loc`
    b_loc = Field(@at loc b)
    filtered_product = Field(filter(Field(a_loc * b_loc)))     # overline(a b)
    return filtered_product - Field(filter(a_loc)) * Field(filter(b_loc))  # − ā b̄
end

end # module
