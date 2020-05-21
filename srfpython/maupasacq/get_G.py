import numpy as np
import matplotlib.pyplot as plt
from sigy.utils.containers import Container
from srfpython.HerrMet.theory import Theory
from srfpython.depthdisp.dispcurves import surf96reader_from_arrays
from srfpython.HerrMet.paramfile import load_paramfile
from srfpython.HerrMet.datacoders import Datacoder_log, makedatacoder
from srfpython.standalone.multipro8 import Job, MapSync, MapAsync
from srfpython.standalone.stdout import waitbarpipe
import scipy.sparse as sp

ny, nx, nz = np.load("nynxnz.npy")
_, _, nper = np.load("nynxnper.npy")

Mprior = np.load("Mprior.npy").reshape((ny, nx, nz))
# phasevel = np.load("phasevel.npy").reshape((ny, nx, nper))
# phasevelunc = np.load("phasevelunc.npy").reshape((ny, nx, nper))
periods = np.load('periods.npy')
ztop = np.load('ztop.npy')
parameterizer_strings = np.load("parameterizer_strings.npy")
datacoder_strings = np.load("datacoder_strings.npy")




# def get_parameterizers(verbose=True, **mapkwargs):
#     """
#     construct parameterizers needed to define the theory function in each node
#     :param self:
#     :return:
#     """
#     # write the header of the parameterizer for each node
#     parameter_string_header = """
#     #met NLAYER = {}
#     #met TYPE = 'mZVSVPvsRHvp'
#     #met VPvs = 'lambda VS: 0.9409+2.0947*VS-0.8206*VS**2+0.2683*VS**3-0.0251*VS**4'
#     #met RHvp = 'lambda VP: 1.6612*VP-0.4721*VP**2+0.0671*VP**3-0.0043*VP**4+0.000106*VP**5'
#     #fld KEY     VINF          VSUP
#     #unt []      []            []
#     #fmt %s      %f            %f
#     """.format(len(ztop)).replace('    #', '#')
#
#     for i in range(1, len(ztop)):
#         # force VINF=VSUP => means lock the depth of the interfaces in the theory operator
#         parameter_string_header += "-Z{} {} {}\n".format(i, -ztop[i], -ztop[i])  # add locked depth interfaces
#
#     def job_generator():
#         for iy in range(ny):
#             for jx in range(nx):# order matters!!!!
#                 vs = Mprior[iy, jx, :]
#                 yield Job(iy, jx, ztop, vs)
#
#     def job_handler(iy, jx, ztop, vs):
#         parameter_string = parameter_string_header
#
#         for i in range(len(ztop)):
#             # SET VINF < VS extracted from pointwise inv < VSUP
#             # such as parameterizer.MMEAN corresponds to the extracted vs
#             parameter_string += "VS{} {} {}\n".format(i, vs[i] - 0.01, vs[i] + 0.01)
#         parameterizer = load_paramfile(parameter_string, verbose=False)[0]
#         return iy, jx, parameterizer
#
#     wb = None
#     if verbose:
#         wb = waitbarpipe('parameterizers')
#
#     parameterizers = []
#     with MapSync(job_handler, job_generator(), **mapkwargs) as ma: # order matters!!!!
#         for jobid, (iy, jx, parameterizer), _, _ in ma:
#             parameterizers.append(parameterizer)
#
#             if verbose:
#                 wb.refresh(jobid / float(nx * ny))
#
#     if verbose:
#         wb.close()
#
#     return parameterizers
#
#



class ForwardOperator(object):
    def __init__(self, verbose=True, **mapkwargs):

        def job_generator():
            for nnode, (parameterizer_string, datacoder_string) in enumerate(zip(
                    parameterizer_strings, datacoder_strings)):
                yield Job(nnode, parameterizer_string, datacoder_string)

        def job_handler(nnode, parameterizer_string, datacoder_string):
            parameterizer = load_paramfile(parameterizer_string, verbose=False)[0]
            datacoder = makedatacoder(datacoder_string, which=Datacoder_log)

            theory = Theory(parameterizer=parameterizer, datacoder=datacoder)

            return nnode, theory

        if verbose:
            wb = waitbarpipe('build g')

        theorys = []
        with MapSync(job_handler, job_generator(), **mapkwargs) as ma:
            for jobid, (nnode, theory), _, _ in ma:
                theorys.append(theory)

                if verbose:
                    wb.refresh(jobid / float(len(parameterizer_strings)))

        if verbose:
            wb.close()

        self.theorys = np.array(theorys, dtype=object).reshape((ny, nx))

    def __call__(self, M, verbose=True, **mapkwargs):
        M = M.reshape((ny, nx, nz))
        theorys = self.theorys

        def job_generator():
            for iy in range(ny):
                for jx in range(nx):  # order matters!!!!
                    nnode = iy * nx + jx
                    yield Job(nnode, theorys[iy, jx], M[iy, jx, :])

        def job_handler(nnode, theory, m):
            data = theory(m=m)
            # print (theory.datacoder.freqs)
            # print (theory.datacoder.values)
            # print (data)
            # print (theory.datacoder.inv(data))
            # print()
            # if nnode == 10:
            #     plt.figure()
            #     plt.plot(theory.datacoder.values)
            #     plt.plot(theory.datacoder.inv(data))
            #     plt.show()
            #     raise

            return nnode, data

        if verbose:
            wb = waitbarpipe('forward pb')
        Data = []
        with MapSync(job_handler, job_generator(), **mapkwargs) as ma:
            for nnode, (nnode, data), _, _ in ma:
                Data.append(data)

                if verbose:
                    wb.refresh(nnode / float(nx * ny))
        if verbose:
            wb.close()

        return np.hstack(Data)  # warning : Data means encoded data

    def frechet_derivatives(self, M, verbose=True, **mapkwargs):

        M = M.reshape((ny, nx, nz))
        theorys = self.theorys

        def job_generator():
            for iy in range(ny):
                for jx in range(nx): # order matters!!!!
                    nnode = iy * nx + jx
                    yield Job(nnode, theorys[iy, jx], M[iy, jx, :])

        def job_handler(nnode, theory, m):
            fd = theory.frechet_derivatives(m=m)
            return nnode, fd

        if verbose:
            wb = waitbarpipe('frechet derivatives')

        # FDs = []
        rows = []
        cols = []
        dats = []
        with MapSync(job_handler, job_generator(), **mapkwargs) as ma:  # order matters
            for jobid, (nnode, fd), _, _ in ma:

                # FDs.append(fd)

                _cols, _rows = np.meshgrid(np.arange(nz), np.arange(nper))
                cols += list(nnode * nz + _cols.flat[:])
                rows += list(nnode * nper + _rows.flat[:])
                dats += list(fd.flat[:])

                if verbose:
                    wb.refresh(nnode / float(nx * ny))

        if verbose:
            wb.close()

        G = sp.csc_matrix((dats, (rows, cols)), shape=(ny*nx*nper, ny*nx*nz), dtype=float)
        return G



if __name__ == '__main__':

    g = ForwardOperator()
    M0 = Mprior.flat[:].copy()
    D0 = g(M0)

    print(D0.shape)
    Dobs = np.load('Dobs.npy')
    print(Dobs.shape)
    plt.plot(Dobs[:1000])
    plt.plot(D0[:1000])
    plt.show()
    # G0 = g.frechet_derivatives(M0)

    # plt.plot(Dobs.flat[:1000])
    # plt.plot(D0[:1000])
    # plt.show()
    # plt.figure()
    # plt.imshow(G0[:1000, :1000].A)
    # plt.show()












#
#
#
# for ix in range(nx):
#     for iy in range(ny):
#         m = Mprior[iy, ix, :]
#         dobs = Dobs[iy, ix, :]
#
#         print (ix, iy, m.shape)
#         print (ix, iy, dobs.shape, periods.shape)