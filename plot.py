#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

x = np.genfromtxt("x.csv")
y = np.genfromtxt("y.csv")
u = np.genfromtxt("u.csv") + 1e-6
v = np.genfromtxt("v.csv") + 1e-6

# Correct indexing
uu = np.flipud(np.rot90(u))
vv = np.flipud(np.rot90(v))
# pp = flipud(rot90(p))

## Plotting
ff.create_streamline(x,y,uu,vv,arrow_scale=.01,density=2).show()
# plt.show()
    # set(streams,'LineWidth',1,'color',[0 0 0])
    # daspect([1 1 1])
    # xlim([0 nx])
    # xlabel('x','fontweight','bold')
    # ylim([0 ny])
    # ylabel('y','fontweight','bold')
    # axis tight
    # box on
