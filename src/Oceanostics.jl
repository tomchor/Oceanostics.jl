module Oceanostics
using DocStringExtensions
using Oceananigans.AbstractOperations: KernelFunctionOperation

const CustomKFO{F} = KernelFunctionOperation{<:Any, <:Any, <:Any, <:Any, <:Any, F}

#+++ Module export
export TracerEquation, KineticEnergyEquation, CoarseGrainedKineticEnergyEquation, TurbulentKineticEnergyEquation, TracerVarianceEquation, PotentialEnergyEquation,
       UMomentumEquation, VMomentumEquation, WMomentumEquation
#---

#+++ TracerEquation exports
export TracerAdvection, TracerDiffusion, TracerImmersedDiffusion, TracerTotalDiffusion, TracerForcing
#---

#+++ UMomentumEquation exports
export UAdvection, UBuoyancyAcceleration, UCoriolisAcceleration, UPressureGradient, UBarotropicPressureGradient,
       UViscousDissipation, UImmersedViscousDissipation, UTotalViscousDissipation,
       UStokesShear, UStokesTendency, UForcing, UTendency
#---

#+++ VMomentumEquation exports
export VAdvection, VBuoyancyAcceleration, VCoriolisAcceleration, VPressureGradient, VBarotropicPressureGradient,
       VViscousDissipation, VImmersedViscousDissipation, VTotalViscousDissipation,
       VStokesShear, VStokesTendency, VForcing, VTendency
#---

#+++ WMomentumEquation exports
export WAdvection, WBuoyancyAcceleration, WCoriolisAcceleration,
       WViscousDissipation, WImmersedViscousDissipation, WTotalViscousDissipation,
       WStokesShear, WStokesTendency, WForcing, WTendency
#---

#+++ TracerVarianceEquation exports
export TracerVarianceTendency, TracerVarianceDissipationRate, TracerVarianceDiffusion
#---

#+++ KineticEnergyEquation exports
export KineticEnergyForcing, KineticEnergyPressureRedistribution, KineticEnergyBuoyancyProduction,
       KineticEnergyDissipationRate, KineticEnergyIsotropicDissipationRate
#---

#+++ CoarseGrainedKineticEnergyEquation exports
export SubfilterStressTensor, KineticEnergyCrossScaleFlux
#---

#+++ TurbulentKineticEnergyEquation exports
export TurbulentKineticEnergy,
       TurbulentKineticEnergyIsotropicDissipationRate,
       TurbulentKineticEnergyXShearProductionRate,
       TurbulentKineticEnergyYShearProductionRate,
       TurbulentKineticEnergyZShearProductionRate,
       TurbulentKineticEnergyShearProductionRate
#---

#+++ FlowDiagnostics exports
export RichardsonNumber, RossbyNumber
export ErtelPotentialVorticity, ThermalWindPotentialVorticity
export DirectionalErtelPotentialVorticity
export StrainRateTensor, StrainRateTensorModulus
export VorticityTensor, VorticityTensorModulus
export Q, QVelocityGradientTensorInvariant
export StressTensor
export subfilter_covariance
export MixedLayerDepth, BuoyancyAnomalyCriterion, DensityAnomalyCriterion
export BottomCellValue
#---

#+++ Filters exports
export BoxFilter, GaussianFilter
#---

#+++ PotentialEnergyEquationTerms exports
export PotentialEnergy
#---

#+++ ProgressMessengers
export ProgressMessengers
#---

#+++ Utils for validation
# Right now, all kernels must be located at ccc
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, ThreeDimensionalFormulation
using Oceananigans.Grids: Center, Face

validate_location(location, type, valid_location=(Center, Center, Center)) =
    location != valid_location &&
        throw(ArgumentError("$type only supports location = $valid_location for now."))

validate_dissipative_closure(closure) = error("Cannot calculate dissipation rate for $closure")
validate_dissipative_closure(::AbstractScalarDiffusivity{<:Any, ThreeDimensionalFormulation}) = nothing
validate_dissipative_closure(closure_tuple::Tuple) = Tuple(validate_dissipative_closure(c) for c in closure_tuple)
#---

#+++ Utils for background fields
using Oceananigans.Fields: Field, ZeroField
"""
    $(SIGNATURES)

Add background fields (velocities and tracers only) to their perturbations.
"""
function add_background_fields(model)

    velocities = model.velocities
    # Adds background velocities to their perturbations only if background velocity isn't ZeroField
    full_velocities = NamedTuple{keys(velocities)}((model.background_fields.velocities[key] isa ZeroField) ?
                                                    val :
                                                    Field(val + model.background_fields.velocities[key])
                                                    for (key, val) in zip(keys(velocities), velocities))
    tracers = model.tracers
    # Adds background tracer fields to their perturbations only if background tracer field isn't ZeroField
    full_tracers = NamedTuple{keys(tracers)}((model.background_fields.tracers[key] isa ZeroField) ?
                                              val :
                                              Field(val + model.background_fields.tracers[key])
                                              for (key, val) in zip(keys(tracers), tracers))

    return merge(full_velocities, full_tracers)
end
#---

#+++ Utils for removing mean fields
using Oceananigans: prognostic_fields, HydrostaticFreeSurfaceModel
using Oceananigans.Biogeochemistry: biogeochemical_auxiliary_fields
"""
    $(SIGNATURES)

Remove mean fields from the model resolved fields.
"""
function perturbation_fields(model; kwargs...)

    resolved_fields = prognostic_fields(model)
    if model isa HydrostaticFreeSurfaceModel
        resolved_fields = (; resolved_fields..., w=ZeroField())
    end

    mean_fields = values(kwargs)
    # Removes mean fields only if mean field is provided
    pert_fields = NamedTuple{keys(resolved_fields)}(haskey(mean_fields, key) ?
                                                    Field(val - mean_fields[key]) :
                                                    val
                                                    for (key,val) in zip(keys(resolved_fields), resolved_fields))
    return merge(pert_fields,
                 model.auxiliary_fields,
                 biogeochemical_auxiliary_fields(model.biogeochemistry))
end
#---

#+++ Utils for Coriolis frequency
using Oceananigans: FPlane, ConstantCartesianCoriolis, AbstractModel
function get_coriolis_frequency_components(coriolis)
    if coriolis isa FPlane
        fx = fy = 0
        fz = coriolis.f
    elseif coriolis isa ConstantCartesianCoriolis
        fx = coriolis.fx
        fy = coriolis.fy
        fz = coriolis.fz
    elseif coriolis == nothing
        fx = fy = fz = 0
    else
        throw(ArgumentError("Extraction of rotation components is only implemented for `FPlane`, `ConstantCartesianCoriolis` and `nothing`."))
    end
    return fx, fy, fz
end

get_coriolis_frequency_components(model::AbstractModel) = get_coriolis_frequency_components(model.coriolis)
#---

#+++ A few utils for closure tuples:
using Oceananigans.TurbulenceClosures: νᶜᶜᶜ

# Fallbacks that capture "single closure" case
@inline _νᶜᶜᶜ(args...) = νᶜᶜᶜ(args...)

# End point
@inline _νᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple{}, K::Tuple{}, clock) = zero(grid)

# Unroll the loop over a tuple
@inline _νᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple, K::Tuple, clock) = νᶜᶜᶜ(i, j, k, grid, closure_tuple[1],     K[1],     clock) +
                                                                     _νᶜᶜᶜ(i, j, k, grid, closure_tuple[2:end], K[2:end], clock)
#---

include("TracerEquation.jl")
include("UMomentumEquation.jl")
include("VMomentumEquation.jl")
include("WMomentumEquation.jl")
include("TracerVarianceEquation.jl")
include("KineticEnergyEquation.jl")
include("TurbulentKineticEnergyEquation.jl")
include("PotentialEnergyEquation.jl")
include("FlowDiagnostics.jl")
include("Filters/Filters.jl")
include("CoarseGrainedKineticEnergyEquation.jl") # depends on FlowDiagnostics (tensors) and Filters
include("ProgressMessengers/ProgressMessengers.jl")

using .TracerEquation, .UMomentumEquation, .VMomentumEquation, .WMomentumEquation, .TracerVarianceEquation, .KineticEnergyEquation, .TurbulentKineticEnergyEquation, .PotentialEnergyEquation
using .FlowDiagnostics
using .Filters
using .CoarseGrainedKineticEnergyEquation
using .ProgressMessengers

#+++ Custom `show` for diagnostics
# Every diagnostic is a `KernelFunctionOperation` (KFO) under the hood, so by default it prints as the
# generic `KernelFunctionOperation at (…)`. Here we give each diagnostic alias its own header
# (`<Name> KernelFunctionOperation at (…)`, so it stays clear the object *is* still a KFO) plus a
# one-line description of what it computes. On color-capable streams the leading `<Name>` and the
# description are tinted with `DESCRIPTION_CRAYON`; everything else (grid, kernel_function, arguments)
# is delegated to Oceananigans' KFO `show`, so we don't duplicate (or drift from) its tree layout.
import Oceananigans.AbstractOperations: operation_name
using Crayons
export DESCRIPTION_CRAYON, set_description_color!

const DESCRIPTION_CRAYON = Ref(crayon"#00BFFF")

"""
    set_description_color!(c::Crayon)

Configure the [`Crayon`](https://github.com/KristofferC/Crayons.jl) used to color a diagnostic's
`show` output (the header's diagnostic name and the `└── computes: …` description line). Build a
`Crayon` with `Crayons.jl` (`using Crayons`), e.g. `crayon"red"` or `Crayon(foreground = :red)`.

```jldoctest
julia> using Oceanostics, Crayons

julia> set_description_color!(crayon"magenta");

julia> DESCRIPTION_CRAYON[] == crayon"magenta"
true
```
"""
set_description_color!(c::Crayon) = (DESCRIPTION_CRAYON[] = c)

"""
Print Oceananigans' KFO tree for `op`, then a `└── computes: …` line with the diagnostic's
`description`. On color-capable streams the leading `name` in the header and the description are tinted
with `DESCRIPTION_CRAYON`. The tree is rendered through Oceananigans' own KFO `show` (into a buffer) so
its layout is reused verbatim; we only recolor the diagnostic name, which is the header's prefix.
"""
function _show_diagnostic(io::IO, op, name, description)
    buf = IOBuffer()
    invoke(show, Tuple{IO, KernelFunctionOperation}, IOContext(buf, :color => false), op)
    body = String(take!(buf))
    colored = get(io, :color, false)
    if colored
        c = DESCRIPTION_CRAYON[]
        body = replace(body, name => string(c, name, inv(c)); count = 1) # header starts with `name`
    end
    print(io, body)
    isempty(description) && return nothing
    if colored
        c = DESCRIPTION_CRAYON[]
        print(io, "\n└── ", c, "computes: ", description, inv(c))
    else
        print(io, "\n└── computes: ", description)
    end
    return nothing
end

"""
Print the compact `summary(op)` one-liner (used when a diagnostic appears nested in a container, e.g.
the `NamedTuple` of components a tensor returns), tinting the leading `name` with `DESCRIPTION_CRAYON`
on color-capable streams — so the name matches the colored header of the multi-line `show`. The
summary starts with `name`, so we recolor only that prefix and leave the rest (` KernelFunctionOperation
at …`) untinted, exactly as the tree does.
"""
function _show_diagnostic_summary(io::IO, op, name)
    s = summary(op)
    if get(io, :color, false)
        c = DESCRIPTION_CRAYON[]
        s = replace(s, name => string(c, name, inv(c)); count = 1)
    end
    print(io, s)
    return nothing
end

"""
    @diagnostic_show T name [description]

Give the `CustomKFO` type alias `T` a custom display: rename its `show`/`summary` header to
`"<name> KernelFunctionOperation"`, and (when `description` is non-empty) append a
`└── computes: <description>` line to its multi-line `show`. On color-capable streams the leading
`<name>` and the description are tinted with `DESCRIPTION_CRAYON`. The header rename goes
through Oceananigans' `operation_name`, so it also propagates (uncolored) to `summary(op)` and to
`op`'s appearance inside larger operation trees and `Field` operand lines.

Following the Julia convention, the full multi-line tree is the *three-arg* `show` (used by the
REPL/`display` for a standalone object), while the *two-arg* `show` is the compact `summary(op)`
one-liner with the leading `<name>` tinted the same `DESCRIPTION_CRAYON` as the tree's header. This
keeps a diagnostic readable when it appears nested in a container — e.g. inside the `NamedTuple` of
components a tensor diagnostic returns, which would otherwise print one full tree per component.
"""
macro diagnostic_show(T, name, description="")
    T = esc(T)
    quote
        # `operation_name` is escaped so the method extends the function imported from Oceananigans
        # (hygiene would otherwise treat the unqualified name as a fresh, separate binding).
        $(esc(:operation_name))(::$T) = $name * " KernelFunctionOperation"
        Base.show(io::IO, ::MIME"text/plain", op::$T) = _show_diagnostic(io, op, $name, $description)
        Base.show(io::IO, op::$T) = _show_diagnostic_summary(io, op, $name)
    end
end

#+++ TracerEquation
@diagnostic_show TracerEquation.Advection         "TracerAdvection"          "tracer advection  ∂ⱼ(uⱼc)"
@diagnostic_show TracerEquation.Diffusion         "TracerDiffusion"          "tracer diffusion (interior)  ∂ⱼqᶜⱼ"
@diagnostic_show TracerEquation.ImmersedDiffusion "TracerImmersedDiffusion"  "tracer diffusion through immersed boundaries  ∂ⱼ𝓆ᶜⱼ"
@diagnostic_show TracerEquation.TotalDiffusion    "TracerTotalDiffusion"     "total tracer diffusion (interior + immersed)  ∂ⱼqᶜⱼ + ∂ⱼ𝓆ᶜⱼ"
#---

#+++ UMomentumEquation
@diagnostic_show UMomentumEquation.Advection                  "UAdvection"                  "advection of u-momentum  ∂ⱼ(uⱼu)"
@diagnostic_show UMomentumEquation.BuoyancyAcceleration       "UBuoyancyAcceleration"       "buoyancy acceleration (x)  ĝₓ b"
@diagnostic_show UMomentumEquation.CoriolisAcceleration       "UCoriolisAcceleration"       "Coriolis acceleration (x)  (f⃗ × u⃗)ₓ"
@diagnostic_show UMomentumEquation.PressureGradient           "UPressureGradient"           "hydrostatic pressure gradient (x)  ∂p/∂x"
@diagnostic_show UMomentumEquation.BarotropicPressureGradient "UBarotropicPressureGradient" "barotropic pressure gradient (x)  g ∂η/∂x"
@diagnostic_show UMomentumEquation.ViscousDissipation         "UViscousDissipation"         "viscous term (interior, x)  ∂ⱼτ₁ⱼ"
@diagnostic_show UMomentumEquation.ImmersedViscousDissipation "UImmersedViscousDissipation" "viscous term through immersed boundaries (x)  ∂ⱼτ₁ⱼ"
@diagnostic_show UMomentumEquation.TotalViscousDissipation    "UTotalViscousDissipation"    "total viscous term (interior + immersed, x)  ∂ⱼτ₁ⱼ"
@diagnostic_show UMomentumEquation.StokesShear                "UStokesShear"                "Stokes shear forcing (x)  ((∇ × u⃗ˢ) × u⃗)ₓ"
@diagnostic_show UMomentumEquation.StokesTendency             "UStokesTendency"             "Stokes drift tendency (x)  ∂uˢ/∂t"
@diagnostic_show UMomentumEquation.Tendency                   "UTendency"                   "total tendency of the u-momentum equation"
#---

#+++ VMomentumEquation
@diagnostic_show VMomentumEquation.Advection                  "VAdvection"                  "advection of v-momentum  ∂ⱼ(uⱼv)"
@diagnostic_show VMomentumEquation.BuoyancyAcceleration       "VBuoyancyAcceleration"       "buoyancy acceleration (y)  ĝ_y b"
@diagnostic_show VMomentumEquation.CoriolisAcceleration       "VCoriolisAcceleration"       "Coriolis acceleration (y)  (f⃗ × u⃗)_y"
@diagnostic_show VMomentumEquation.PressureGradient           "VPressureGradient"           "hydrostatic pressure gradient (y)  ∂p/∂y"
@diagnostic_show VMomentumEquation.BarotropicPressureGradient "VBarotropicPressureGradient" "barotropic pressure gradient (y)  g ∂η/∂y"
@diagnostic_show VMomentumEquation.ViscousDissipation         "VViscousDissipation"         "viscous term (interior, y)  ∂ⱼτ₂ⱼ"
@diagnostic_show VMomentumEquation.ImmersedViscousDissipation "VImmersedViscousDissipation" "viscous term through immersed boundaries (y)  ∂ⱼτ₂ⱼ"
@diagnostic_show VMomentumEquation.TotalViscousDissipation    "VTotalViscousDissipation"    "total viscous term (interior + immersed, y)  ∂ⱼτ₂ⱼ"
@diagnostic_show VMomentumEquation.StokesShear                "VStokesShear"                "Stokes shear forcing (y)  ((∇ × u⃗ˢ) × u⃗)_y"
@diagnostic_show VMomentumEquation.StokesTendency             "VStokesTendency"             "Stokes drift tendency (y)  ∂vˢ/∂t"
@diagnostic_show VMomentumEquation.Tendency                   "VTendency"                   "total tendency of the v-momentum equation"
#---

#+++ WMomentumEquation
@diagnostic_show WMomentumEquation.Advection                  "WAdvection"                  "advection of w-momentum  ∂ⱼ(uⱼw)"
@diagnostic_show WMomentumEquation.BuoyancyAcceleration       "WBuoyancyAcceleration"       "buoyancy acceleration (z)  ĝ_z b"
@diagnostic_show WMomentumEquation.CoriolisAcceleration       "WCoriolisAcceleration"       "Coriolis acceleration (z)  (f⃗ × u⃗)_z"
@diagnostic_show WMomentumEquation.ViscousDissipation         "WViscousDissipation"         "viscous term (interior, z)  ∂ⱼτ₃ⱼ"
@diagnostic_show WMomentumEquation.ImmersedViscousDissipation "WImmersedViscousDissipation" "viscous term through immersed boundaries (z)  ∂ⱼτ₃ⱼ"
@diagnostic_show WMomentumEquation.TotalViscousDissipation    "WTotalViscousDissipation"    "total viscous term (interior + immersed, z)  ∂ⱼτ₃ⱼ"
@diagnostic_show WMomentumEquation.StokesShear                "WStokesShear"                "Stokes shear forcing (z)  ((∇ × u⃗ˢ) × u⃗)_z"
@diagnostic_show WMomentumEquation.StokesTendency             "WStokesTendency"             "Stokes drift tendency (z)  ∂wˢ/∂t"
@diagnostic_show WMomentumEquation.Tendency                   "WTendency"                   "total tendency of the w-momentum equation"
#---

#+++ TracerVarianceEquation
@diagnostic_show TracerVarianceEquation.Tendency        "TracerVarianceTendency"        "tracer variance tendency  2c ∂ₜc"
@diagnostic_show TracerVarianceEquation.Diffusion       "TracerVarianceDiffusion"       "tracer variance diffusion  2c ∂ⱼFⱼ"
@diagnostic_show TracerVarianceEquation.DissipationRate "TracerVarianceDissipationRate" "tracer variance dissipation rate  χ = 2 ∂ⱼc·Fⱼ"
#---

#+++ KineticEnergyEquation
@diagnostic_show KineticEnergyEquation.KineticEnergy                         "KineticEnergy"                       "kinetic energy  ½uᵢuᵢ"
@diagnostic_show KineticEnergyEquation.KineticEnergyTendency                 "KineticEnergyTendency"               "kinetic energy tendency  uᵢGᵢ (excl. nonhydrostatic pressure)"
@diagnostic_show KineticEnergyEquation.KineticEnergyAdvection                "KineticEnergyAdvection"              "kinetic energy advection  uᵢ∂ⱼ(uᵢuⱼ)"
@diagnostic_show KineticEnergyEquation.KineticEnergyStress                   "KineticEnergyStress"                 "kinetic energy stress/diffusion  uᵢ∂ⱼτᵢⱼ"
@diagnostic_show KineticEnergyEquation.KineticEnergyForcing                  "KineticEnergyForcing"                "kinetic energy forcing  uᵢFᵤᵢ"
@diagnostic_show KineticEnergyEquation.KineticEnergyPressureRedistribution   "KineticEnergyPressureRedistribution" "kinetic energy pressure redistribution  uᵢ∂ᵢp"
@diagnostic_show KineticEnergyEquation.KineticEnergyBuoyancyProduction       "KineticEnergyBuoyancyProduction"     "kinetic energy buoyancy production  uᵢbᵢ"
@diagnostic_show KineticEnergyEquation.KineticEnergyDissipationRate          "KineticEnergyDissipationRate"        "kinetic energy dissipation rate  ε = ∂ⱼuᵢ·Fᵢⱼ"
@diagnostic_show KineticEnergyEquation.KineticEnergyIsotropicDissipationRate "KineticEnergyIsotropicDissipationRate" "isotropic kinetic energy dissipation rate  ε = 2νSᵢⱼSᵢⱼ"
#---

#+++ CoarseGrainedKineticEnergyEquation
# Only the (single-`KernelFunctionOperation`) flux gets a custom display; `SubfilterStressTensor`
# returns a `NamedTuple` of components, like `StressTensor`/`StrainRateTensor`, so it has none.
@diagnostic_show CoarseGrainedKineticEnergyEquation.KineticEnergyCrossScaleFlux "KineticEnergyCrossScaleFlux" "cross-scale kinetic energy flux  Πₖ = -τⁱʲS̄ⁱʲ"
#---

#+++ TurbulentKineticEnergyEquation
@diagnostic_show TurbulentKineticEnergyEquation.TurbulentKineticEnergy                     "TurbulentKineticEnergy"                     "turbulent kinetic energy  ½uᵢ′uᵢ′"
@diagnostic_show TurbulentKineticEnergyEquation.TurbulentKineticEnergyXShearProductionRate "TurbulentKineticEnergyXShearProductionRate" "TKE shear production (x)  -uᵢ′u′ ∂ₓUᵢ"
@diagnostic_show TurbulentKineticEnergyEquation.TurbulentKineticEnergyYShearProductionRate "TurbulentKineticEnergyYShearProductionRate" "TKE shear production (y)  -uᵢ′v′ ∂_yUᵢ"
@diagnostic_show TurbulentKineticEnergyEquation.TurbulentKineticEnergyZShearProductionRate "TurbulentKineticEnergyZShearProductionRate" "TKE shear production (z)  -uᵢ′w′ ∂_zUᵢ"
@diagnostic_show TurbulentKineticEnergyEquation.TurbulentKineticEnergyShearProductionRate  "TurbulentKineticEnergyShearProductionRate"  "total TKE shear production  -uᵢ′uⱼ′ ∂ⱼUᵢ"
#---

#+++ PotentialEnergyEquation
@diagnostic_show PotentialEnergyEquation.PotentialEnergy "PotentialEnergy" "potential energy per unit volume  Eₚ = -bz"
#---

#+++ FlowDiagnostics
@diagnostic_show FlowDiagnostics.RichardsonNumber                   "RichardsonNumber"                   "Richardson number  Ri = (∂b/∂z) / |∂u⃗ₕ/∂z|²"
@diagnostic_show FlowDiagnostics.RossbyNumber                       "RossbyNumber"                       "Rossby number  Ro = ωᶻ/f"
@diagnostic_show FlowDiagnostics.ErtelPotentialVorticity            "ErtelPotentialVorticity"            "Ertel potential vorticity  q = ω⃗ₜₒₜ · ∇b"
@diagnostic_show FlowDiagnostics.ThermalWindPotentialVorticity      "ThermalWindPotentialVorticity"      "Ertel PV, thermal-wind form  q = (f + ωᶻ)∂b/∂z - f((∂U/∂z)² + (∂V/∂z)²)"
@diagnostic_show FlowDiagnostics.DirectionalErtelPotentialVorticity "DirectionalErtelPotentialVorticity" "directional contribution to Ertel PV  (f̂ + ω̂)·∇b along a direction"
@diagnostic_show FlowDiagnostics.StrainRateTensorModulus            "StrainRateTensorModulus"            "strain-rate tensor modulus  √(SᵢⱼSᵢⱼ)"
@diagnostic_show FlowDiagnostics.VorticityTensorModulus             "VorticityTensorModulus"             "vorticity tensor modulus  √(ΩᵢⱼΩᵢⱼ)"
@diagnostic_show FlowDiagnostics.QVelocityGradientTensorInvariant   "QVelocityGradientTensorInvariant"   "Q velocity-gradient invariant  Q = ½(ΩᵢⱼΩᵢⱼ - SᵢⱼSᵢⱼ)"
@diagnostic_show FlowDiagnostics.StrainRateTensor                   "StrainRateTensor"                   "strain-rate tensor component  Sᵢⱼ = ½(∂ⱼuᵢ + ∂ᵢuⱼ)"
@diagnostic_show FlowDiagnostics.VorticityTensor                    "VorticityTensor"                    "vorticity tensor component  Ωᵢⱼ = ½(∂ⱼuᵢ - ∂ᵢuⱼ)"
@diagnostic_show FlowDiagnostics.StressTensor                       "StressTensor"                       "stress tensor component  τᵢⱼ = uᵢuⱼ"
@diagnostic_show CustomKFO{<:FlowDiagnostics.MixedLayerDepthKernel} "MixedLayerDepth"                    "mixed layer depth (shallowest depth where the criterion is met)"
#---

#+++ Filters
@diagnostic_show Filters.BoxFilter      "BoxFilter"      "local box-average (running mean) of the operand"
@diagnostic_show Filters.GaussianFilter "GaussianFilter" "local Gaussian-weighted average of the operand"
#---
#---

end # module
