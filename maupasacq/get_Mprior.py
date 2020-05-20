import numpy as np
import matplotlib.pyplot as plt
from sigy.utils.containers import Container

cont = Container()
cont.loadkeys('maupasacq_swavevelocity_cube.npz',
              ['xrot_km',
               'yrot_km',
               'ztop_km',
               'vs_km_per_sec',
               'vsunc_km_per_sec'])

xrot = cont['xrot_km']
yrot = cont['yrot_km']
ztop = cont['ztop_km']
vs_cube = cont['vs_km_per_sec']
vsunc_cube = cont['vsunc_km_per_sec']

# =============== down sample z axis
# current config = z, y, x
from srfpython.depthdisp.depthmodels import depthmodel1D
new_ztop = np.linspace(0., 10., 11)
new_zmid = np.hstack((new_ztop[:-1] + 0.5 * (new_ztop[1:] - new_ztop[:-1]), new_ztop[-1] + new_ztop[1] - new_ztop[0]))
vs_ij = depthmodel1D(ztop=ztop, values=ztop * 0. + 1.)
new_vs_cube = np.zeros((len(new_ztop), len(yrot), len(xrot)), float)
new_vsunc_cube = np.zeros((len(new_ztop), len(yrot), len(xrot)), float)

for iy in range(len(yrot)):
    for jx in range(len(xrot)):
        # plt.gca().cla()

        vs_ij.values[:] = vs_cube[:, iy, jx]
        new_vs_cube[:, iy, jx] = vs_ij.interp(z=new_zmid)
        # vs_ij.show(plt.gca())

        vs_ij.values[:] += vsunc_cube[:, iy, jx]
        new_vs_sup = vs_ij.interp(z=new_zmid)
        # vs_ij.show(plt.gca())

        vs_ij.values[:] -= 2 * vsunc_cube[:, iy, jx]
        new_vs_inf = vs_ij.interp(z=new_zmid)
        # vs_ij.show(plt.gca())

        new_vsunc_cube[:, iy, jx] = 0.5 * (new_vs_sup - new_vs_inf)

        # new_vs = depthmodel1D(ztop=new_ztop, values=new_vs)
        # new_vs.show(plt.gca(), "--")
        #
        # new_vs.values[:] += new_vs_unc
        # new_vs.show(plt.gca(), "--")
        #
        # new_vs.values[:] -= 2 * new_vs_unc
        # new_vs.show(plt.gca(), "--")
        #
        # plt.ion()
        # plt.show()
        # raw_input('')
        # plt.pause(0.1)

del ztop, vs_cube, vsunc_cube
ztop = new_ztop
zmid=  new_zmid
vs_cube = new_vs_cube
vsunc_cube = new_vsunc_cube

# ===============
# change dimension order from z, y, x to y, x, z
# print(vs_cube.shape)
iz, jy, kx = np.mgrid[:len(ztop), :len(yrot), :len(xrot)]

x = xrot[kx].swapaxes(0, 2).swapaxes(0, 1).flat[:]
y = yrot[jy].swapaxes(0, 2).swapaxes(0, 1).flat[:]
z = ztop[iz].swapaxes(0, 2).swapaxes(0, 1).flat[:]
Mprior = vs_cube.swapaxes(0, 2).swapaxes(0, 1).flat[:]
Munc = vsunc_cube.swapaxes(0, 2).swapaxes(0, 1).flat[:]

np.save('nynxnz.npy', np.array([len(yrot), len(xrot), len(ztop)]))
np.save('x.npy', x)
np.save('y.npy', y)
np.save('z.npy', z)
np.save('Mprior.npy', Mprior)
np.save('Munc.npy', Munc)