```@contents
Pages = ["library.md"]
```

# Library

Documentation for the public user interface.

```@autodocs
Modules = [Oceanostics,
           Oceanostics.TracerEquation,
           Oceanostics.TracerVarianceEquation,
           Oceanostics.TurbulentKineticEnergyEquation,
           Oceanostics.KineticEnergyEquation,
           Oceanostics.FlowDiagnostics,
           Oceanostics.PotentialEnergyEquation,
           Oceanostics.ProgressMessengers]
Order = [:function, :type, :macro]
Filter = t -> true
```
