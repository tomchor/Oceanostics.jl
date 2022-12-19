import xarray as xr
import pynanigans as pn
from matplotlib import pyplot as plt

grid, ds = pn.open_simulation("tracer_diff.nc", load=True, decode_times=False)

c2_zint = grid.integrate(ds.c2_int, "z")
c2_zint -= c2_zint.sel(time=0)

χ_zint = grid.integrate(ds.χ_int, "z")
χ_tzint = -pn.regular_indef_integrate(χ_zint, "time")
χ_tzint -= χ_tzint.sel(time=0)


ε_q_zint = grid.integrate(ds.ε_q_int, "z")
ε_q_tzint = -pn.regular_indef_integrate(ε_q_zint, "time")
ε_q_tzint -= ε_q_tzint.sel(time=0)

dc2dt = c2_zint.differentiate("time")

fig, axes = plt.subplots(nrows=2, constrained_layout=True)

c2_zint.plot(ax=axes[0], x="time", label="c²")
χ_tzint.plot(ax=axes[0], x="time", label="∫χdt")
#ε_q_tzint.plot(ax=axes[0], x="time", label="∫ε_qdt")

(-dc2dt).plot(ax=axes[1], x="time", label="dc²/dt")
χ_zint.plot(ax=axes[1], x="time", label="χ")
#ε_q_zint.plot(ax=axes[1], x="time", label="ε_q")

for ax in axes:
    ax.legend(); ax.grid(True)
