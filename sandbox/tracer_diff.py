import xarray as xr
import pynanigans as pn
from matplotlib import pyplot as plt

grid, ds = pn.open_simulation("tracer_diff.nc", load=True, decode_times=False)

c2_int = grid.integrate(ds.c2_int, "z")
c2_int -= c2_int.sel(time=0)

χ_int = -pn.regular_indef_integrate(grid.integrate(ds.χ_int, "z"), "time")
χ_int -= χ_int.sel(time=0)

