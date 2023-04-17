import os
import time
import torch as th
import torchbox as tb
import torchsar as ts
from dataset import loaddata
from network import Net, raw2img
import torchcs as tc
import matplotlib

seed = 2021
device = 'cpu'
useCoarsePhi = True
# useCoarsePhi = False
isEasy = True
# isEasy = False

usePsi = True
usePsi = False
method = 'mp'
# method = 'omp'
method = 'fista'
alpha = 4e-5
# alpha = None
outfolder = './snapshot/CS/'
sarmodelfile = '../SARModel/SARModelSim4Size4096x4096.mat'

if isEasy:
    # datafile1500, ntasks, npoints = '/mnt/e/DataSets/zhi/sar/sim/meta/Easy/Points/EasyMetaSARTasks1500Points4SceneSize64x64EchoSize64x64.h5', 1500, 4
    # datafile20, ntasks, npoints = '/mnt/e/DataSets/zhi/sar/sim/meta/Easy/Points/EasyMetaSARTasks20Points4SceneSize64x64EchoSize64x64.h5', 20, 4
    datafile20, ntasks, npoints = '/mnt/e/DataSets/zhi/sar/sim/meta/Easy/PointShapes/EasyMetaSARTasks20Points4SceneSize64x64EchoSize64x64.h5', 20, 4
    # datafile20, ntasks, npoints = '/mnt/e/ws/github/tsar/tsar/examples/datasets/compressed/EasyMetaSARTasks20Points4SceneSize64x64EchoSize64x64.h5', 20, 4
else:
    # datafile1500, ntasks, npoints = '/mnt/e/DataSets/zhi/sar/sim/meta/Hard/Points/HardMetaSARTasks1500Points4SceneSize64x64EchoSize64x64.h5', 1500, 4
    # datafile20, ntasks, npoints = '/mnt/e/DataSets/zhi/sar/sim/meta/Hard/Points/HardMetaSARTasks20Points4SceneSize64x64EchoSize64x64.h5', 20, 4
    datafile20, ntasks, npoints = '/mnt/e/DataSets/zhi/sar/sim/meta/Hard/PointShapes/HardMetaSARTasks20Points4ScatterSize64x64EchoSize64x64.h5', 20, 4

ntasks, npoints = 20, 4
nTest = ntasks * npoints

data = tb.loadh5(datafile20, keys=None)
print(data.keys())
Sr, SiMF, SiFISTA, X = th.from_numpy(data['Sr']), th.from_numpy(data['SiMF']), th.from_numpy(data['SiFISTA']), th.from_numpy(data['X'])
# SiFISTA = th.view_as_complex(SiFISTA)

if useCoarsePhi:
    Phi = th.from_numpy(tb.loadmat(sarmodelfile)['Phi']).contiguous()
    Phi = th.view_as_complex(Phi)

gdshape = [X.shape[1], X.shape[2]]
ES = [Sr.shape[1], Sr.shape[2]]
Psi = tc.idftmtx(gdshape[0] * gdshape[1])
# Psi = tc.idctmtx(gdshape[0] * gdshape[1])
K = int(ES[0] * ES[1] / 4)
N = Sr.shape[0]
idxTest = list(range(0, nTest, int(nTest / 20)))
idxTest = [0, 56]
print(idxTest, "=====")
SIs = th.zeros([len(idxTest)] + gdshape)
SIMFs = th.zeros([len(idxTest)] + gdshape)
Xs = th.zeros([len(idxTest)] + gdshape)
cnt = 0
# for n in range(0, N):
for n in idxTest:
    task = int(n / npoints)
    if not useCoarsePhi:
        # print(pdict['PRF'])
        pdict = data['pdict' + str(task)]
        # pdict['V'] = 152
        # pdict = ts.compute_sar_parameters(pdict, islog=False)
        # print(pdict['PRF'])
        Phi = ts.sarmodel(pdict, mod='2D1', gdshape=gdshape, device=device, islog=False)
        # print(Phi.abs().min(), Phi.abs().max())
    # print(Phi.abs().min(), Phi.abs().max())

    Psi = Psi.to(Phi.dtype)
    PhiH = Phi.t().conj()
    if usePsi:
        A = Phi.mm(Psi)
    else:
        A = Phi

    y = Sr[n]
    x = X[n]

    y = th.view_as_complex(y)
    y = y.reshape(ES[0] * ES[1], 1)
    print(y.shape, A.shape)
    tstart = time.time()
    if method == 'mp':
        z, _ = tc.mp(y, A, K=K, norm=[False, True], tol=1.0e-6, mode='real', islog=False)
    if method == 'omp':
        z, _ = tc.omp(y, A, K=K, C=0.00001, norm=[False, False], tol=1.0e-4, method='pinv', mode='real', islog=False)
    if method == 'fista':
        z = tc.fista(y, A, niter=800, lambd=0.9, alpha=alpha, tol=1.0e-6, ssmode='cc')
    tend = time.time()
    print(tend - tstart)

    # print(z)
    if usePsi:
        SI = Psi.mm(z)
    else:
        SI = z
    SIMF = PhiH.mm(y)
    # print(S.shape, Phi.shape, A.shape)
    # SI = th.pinverse(Phi).mm(S)

    print("---SAR image data shape and dtype: ", SI.shape, SI.dtype)
    print("---Minimum and maximum value in SAR image data: ", SI.real.min(), SI.real.max(), SI.imag.min(), SI.imag.max())
    SIs[cnt] = SI.reshape(gdshape[0], gdshape[1])
    SIMFs[cnt] = SIMF.reshape(gdshape[0], gdshape[1])
    Xs[cnt] = x
    cnt = cnt + 1

SIs = SIs.abs()
SIMFs = SIMFs.abs()
Xs = Xs.abs()

# SIs, SIMFs, Xs = tb.mapping(SIs).numpy(), tb.mapping(SIMFs).numpy(), tb.mapping(Xs).numpy()

for n, SI, SIMF, X in zip(idxTest, SIs, SIMFs, Xs):
    tb.imsave(outfolder + method + 'Image' + str(n) + '.png', SI)
    tb.imsave(outfolder + 'mf' + 'Image' + str(n) + '.png', SIMF)
    tb.imsave(outfolder + 'scatter' + 'Image' + str(n) + '.png', X)
    n = n + 1
    print(n, SI.shape, SIMF.shape, X.shape)
