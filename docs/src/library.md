@meta
EditURL = "<unknown>/library.md"
```

```@contents
Pages = ["library.md"]
```

# Library

Documentation for the public user interface.

```@autodocs
Modules = [Oceanostics,
           Oceanostics.TKEEquation,
           Oceanostics.TracerEquation,
           Oceanostics.TracerVarianceEquation,
           Oceanostics.FlowDiagnostics,
           Oceanostics.PotentialEnergyEquation,
           Oceanostics.ProgressMessengers]
Order = [:function, :type, :macro]
Filter = t -> true
```

## Specific Functions

```@docs
Oceanostics.TKEEquation.KineticEnergyTendency
Oceanostics.FlowDiagnostics.QVelocityGradientTensorInvariant
Oceanostics.FlowDiagnostics.VorticityTensorModulus
Oceanostics.PotentialEnergyEquation.PotentialEnergy
Oceanostics.TKEEquation.TurbulentKineticEnergy
Oceanostics.TKEEquation.KineticEnergyForcingTerm
Oceanostics.TKEEquation.KineticEnergy
Oceanostics.TKEEquation.AdvectionTerm
Oceanostics.TracerEquation.TracerForcing
Oceanostics.FlowDiagnostics.RossbyNumber
Oceanostics.TKEEquation.KineticEnergyDissipationRate
Oceanostics.TracerVarianceEquation.TracerVarianceDiffusiveTerm
Oceanostics.TKEEquation.BuoyancyProductionTerm
Oceanostics.FlowDiagnostics.RichardsonNumber
Oceanostics.FlowDiagnostics.DensityAnomalyCriterion
Oceanostics.FlowDiagnostics.DirectionalErtelPotentialVorticity
Oceanostics.TKEEquation.PressureRedistributionTerm
Oceanostics.FlowDiagnostics.StrainRateTensorModulus
Oceanostics.TKEEquation.ZShearProductionRate
Oceanostics.TKEEquation.YShearProductionRate
Oceanostics.TracerEquation.TracerDiffusion
Oceanostics.TracerVarianceEquation.TracerVarianceDissipationRate
Oceanostics.TracerEquation.ImmersedTracerDiffusion
Oceanostics.FlowDiagnostics.BuoyancyAnomalyCriterion
Oceanostics.TracerEquation.TracerAdvection
Oceanostics.TKEEquation.IsotropicKineticEnergyDissipationRate
Oceanostics.FlowDiagnostics.MixedLayerDepth
Oceanostics.FlowDiagnostics.ErtelPotentialVorticity
Oceanostics.TKEEquation.KineticEnergyStressTerm
Oceanostics.FlowDiagnostics.BottomCellValue
Oceanostics.FlowDiagnostics.ThermalWindPotentialVorticity
Oceanostics.TKEEquation.XShearProductionRate
Oceanostics.TracerEquation.TotalTracerDiffusion
Oceanostics.TracerVarianceEquation.TracerVarianceTendency
```
