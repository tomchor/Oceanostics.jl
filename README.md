# KCF_diagnostics

`KernelComputedField`s diagnostics for use with Oceananigans.

- Not every kernel has been tested.
- Kernels are written very generally since most uses of averages, etc. do not assume any
  specific kind of averaging procedure. Chances are it "wastes" computations.
- For now this isn't meant to be a module, but simply a collection of Kernels that can be
  adapted for specific uses.
