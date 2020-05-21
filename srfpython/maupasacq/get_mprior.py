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

x = cont['xrot_km']
y = cont['yrot_km']
ztop = cont['ztop_km']
vs_cube = cont['vs_km_per_sec']
vsunc_cube = cont['vsunc_km_per_sec']

# =============== down sample z axis
# current config = z, y, x
from srfpython.depthdisp.depthmodels import depthmodel1D
new_ztop = np.linspace(0., 10., 11)
new_zmid = np.hstack((new_ztop[:-1] + 0.5 * (new_ztop[1:] - new_ztop[:-1]), new_ztop[-1] + new_ztop[1] - new_ztop[0]))
vs_ij = depthmodel1D(ztop=ztop, values=ztop * 0. + 1.)
new_vs_cube = np.zeros((len(new_ztop), len(y), len(x)), float)
new_vsunc_cube = np.zeros((len(new_ztop), len(y), len(x)), float)

for iy in range(len(y)):
    for jx in range(len(x)):
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
nz, ny, nx = len(ztop), len(y), len(x)
iz, jy, kx = np.mgrid[:nz, :ny, :nx]

xflat = x[kx].swapaxes(0, 2).swapaxes(0, 1).flat[:]
yflat = y[jy].swapaxes(0, 2).swapaxes(0, 1).flat[:]
ztopflat = ztop[iz].swapaxes(0, 2).swapaxes(0, 1).flat[:]
zmidflat = zmid[iz].swapaxes(0, 2).swapaxes(0, 1).flat[:]
Mprior = vs_cube.swapaxes(0, 2).swapaxes(0, 1).flat[:]
Munc = vsunc_cube.swapaxes(0, 2).swapaxes(0, 1).flat[:]

np.save('nynxnz.npy', np.array([len(y), len(x), len(ztop)]))
np.save('x.npy', x)
np.save('y.npy', y)
np.save('ztop.npy', ztop)
np.save('zmid.npy', ztop)
np.save('xflat.npy', xflat)
np.save('yflat.npy', yflat)
np.save('ztopflat.npy', ztopflat)
np.save('zmidflat.npy', zmidflat)
np.save('Mprior.npy', Mprior)
np.save('Munc.npy', Munc)






from srfpython.HerrMet.paramfile import load_paramfile
from srfpython.standalone.multipro8 import Job, MapSync, MapAsync
from srfpython.standalone.stdout import waitbarpipe

Mprior = Mprior.reshape((ny, nx, nz))


def get_parameterizers(verbose=True, **mapkwargs):
    """
    construct parameterizers needed to define the theory function in each node
    :param self:
    :return:
    """
    # write the header of the parameterizer for each node
    parameter_string_header = """
    #met NLAYER = {}
    #met TYPE = 'mZVSVPvsRHvp'
    #met VPvs = 'lambda VS: 0.9409+2.0947*VS-0.8206*VS**2+0.2683*VS**3-0.0251*VS**4'
    #met RHvp = 'lambda VP: 1.6612*VP-0.4721*VP**2+0.0671*VP**3-0.0043*VP**4+0.000106*VP**5'
    #fld KEY     VINF          VSUP
    #unt []      []            []
    #fmt %s      %f            %f
    """.format(len(ztop)).replace('    #', '#')

    for i in range(1, len(ztop)):
        # force VINF=VSUP => means lock the depth of the interfaces in the theory operator
        parameter_string_header += "-Z{} {} {}\n".format(i, -ztop[i], -ztop[i])  # add locked depth interfaces

    def job_generator():
        for iy in range(ny):
            for jx in range(nx):# order matters!!!!
                vs = Mprior[iy, jx, :]
                yield Job(iy, jx, ztop, vs)

    def job_handler(iy, jx, ztop, vs):
        parameterizer_string = parameter_string_header

        for i in range(len(ztop)):
            # SET VINF < VS extracted from pointwise inv < VSUP
            # such as parameterizer.MMEAN corresponds to the extracted vs
            parameterizer_string += "VS{} {} {}\n".format(i, vs[i] - 0.01, vs[i] + 0.01)
        # parameterizer = load_paramfile(parameter_string, verbose=False)[0]
        return iy, jx, parameterizer_string

    wb = None
    if verbose:
        wb = waitbarpipe('parameterizers')

    parameterizer_strings = []
    with MapSync(job_handler, job_generator(), **mapkwargs) as ma: # order matters!!!!
        for jobid, (iy, jx, parameter_string), _, _ in ma:
            parameterizer_strings.append(parameter_string)

            if verbose:
                wb.refresh(jobid / float(nx * ny))

    if verbose:
        wb.close()

    return np.asarray(parameterizer_strings, str)


parameterizer_strings = get_parameterizers(verbose=True)
np.save('parameterizer_strings.npy', parameterizer_strings)