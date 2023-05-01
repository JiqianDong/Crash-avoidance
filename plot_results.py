import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

from scipy.interpolate import RegularGridInterpolator

data_f = "./runs/speed_20_success_rate2.txt"
data = pd.read_csv(data_f,delimiter=',', header=None)
# "{PLANNING_HORIZON},{NUM_TRAJECORIES},{CEM_ITERS},{sr},{avg_t}"
data.columns = ["planning_horizon", "num_trajecories", "cem_iters", "success_rate", "computation time"]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15,15))
# norm_fn = plt.Normalize()


xs, ys, zs, srs, time = data["planning_horizon"], data["num_trajecories"], \
    data["cem_iters"], data["success_rate"], data["computation time"]


# interp_fn = RegularGridInterpolator((xs, ys, zs), srs)

# X = np.linspace(xs.min(), xs.max(), 100)
# Y = np.linspace(ys.min(), ys.max(), 100)
# Z = np.linspace(zs.min(), zs.max(), 100)
# colors = cmap(norm_fn(srs))
# value = time
value = srs

cmap = mpl.colormaps["RdYlGn"]
# cmap = cmap.reversed()


ax.scatter(xs, ys, zs, c=value, cmap=cmap, s=500)

# for x,y,z,t,c in zip(xs, ys, zs, srs, colors):
#     ax.text(x, y, z, str(t), color=c)
#     ax.bar3d(x, y, z, dx=1, dy=1, dz=1, color=c)
norm = mpl.colors.Normalize(vmin=min(value), vmax=max(value))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
  
plt.colorbar(sm, ticks=np.linspace(min(value), max(value), 5))
ax.set_xlabel("planning horizon")
ax.set_ylabel("num trajecories")
ax.set_zlabel("CEM iterations")
plt.show()






















########################################


# ZZ = np.outer(Z.T, Z)        # 50x50
# XX, YY = np.meshgrid(X, Y)    # 50x50

# print(XX.shape)
# val = interp_fn(XX, YY, ZZ)
# # fourth dimention - colormap
# # create colormap according to x-value (can use any 50x50 array)
# color_dimension = val # change to desired fourth dimension
# minn, maxx = color_dimension.min(), color_dimension.max()
# norm = norm_fn(minn, maxx)
# m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
# m.set_array([])
# fcolors = m.to_rgba(color_dimension)

# # plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()
