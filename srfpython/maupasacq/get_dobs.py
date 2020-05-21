import numpy as np
import matplotlib.pyplot as plt
from sigy.utils.containers import Container
from srfpython.depthdisp.dispcurves import surf96reader_from_arrays
from srfpython.HerrMet.datacoders import Datacoder_log, makedatacoder
from srfpython.standalone.stdout import waitbarpipe


cont = Container()
cont.loadkeys('maupasacq_phasevelocity_cube.npz',
              ['phase_velocity_km_per_sec',
               'period_sec'])

phasevel_cube = cont['phase_velocity_km_per_sec']
periods = cont['period_sec']
nper, ny, nx = phasevel_cube.shape
# ===============
# current config = p, y, x


# ===============
# change dimension order from p, y, x to y, x, p
# print(vs_cube.shape)
ip, jy, kx = np.mgrid[:nper, :ny, :nx]

# x = xrot[kx].swapaxes(0, 2).swapaxes(0, 1).flat[:]
# y = yrot[jy].swapaxes(0, 2).swapaxes(0, 1).flat[:]
pflat = periods[ip].swapaxes(0, 2).swapaxes(0, 1).flat[:]
phasevel = phasevel_cube.swapaxes(0, 2).swapaxes(0, 1)  # .flat[:]  # this is not Dobs
phasevelunc = 0.1 * phasevel  # the uncertainty used for the depth inversion


def get_datacoders(verbose=True):
    datacoder_strings = []

    if verbose:
        wb = waitbarpipe('datacoders    ')

    for iy in range(ny):
        for jx in range(nx): # order matters!!!!
            nnode = iy * nx + jx

            s96 = surf96reader_from_arrays(
                waves=['R' for _ in range(nper)],
                types=['C' for _ in range(nper)],
                modes=np.zeros(nper, int),
                freqs=1./periods,
                values=phasevel[iy, jx, :],
                dvalues=phasevelunc[iy, jx, :])

            datacoder_string = str(s96)
            datacoder_strings.append(datacoder_string)

            if verbose:
                wb.refresh(nnode / float(nx * ny))

    if verbose:
        wb.close()

    datacoder_strings = np.asarray(datacoder_strings, str)
    datacoders = [makedatacoder(datacoder_string, which=Datacoder_log) for datacoder_string in datacoder_strings]
    return datacoders, datacoder_strings


datacoders, datacoder_strings = get_datacoders()
Dobs = []
Dunc = []
for datacoder in datacoders:
    _dobs, _CDinv = datacoder.target()
    Dobs.append(_dobs)
    Dunc.append(_CDinv ** -0.5)

Dobs = np.hstack(Dobs)
Dunc = np.hstack(Dunc)

np.save('datacoder_strings.npy', datacoder_strings)
np.save('nynxnper.npy', np.array([ny, nx, nper]))
np.save('periods.npy', periods)
np.save('pflat.npy', pflat)
np.save('Dobs.npy', Dobs)
np.save('Dunc.npy', Dunc)