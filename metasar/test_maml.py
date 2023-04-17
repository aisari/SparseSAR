import os
import torch as th
import torchbox as tb
from dataset import loaddata
from network import raw2img, Net
from solver import check, fast_adapt
import learn2learn as l2l


device = 'cuda:1'

schname = 'scheme11'
sch = tb.loadyaml('./scheme.yaml', schname)
inchnl, iscplx = (1, True) if sch['nnvtype'] == 'complex' else (2, False)

outfolder = tb.pathjoin('records', schname, 'maml')
os.makedirs(outfolder) if not os.path.exists(outfolder) else 0

print(sch)

SrTest, SiTest, XTest = th.tensor([]), th.tensor([]), th.tensor([])
for testname, nstest in zip(sch['dstest']['name'], sch['dstest']['ntasks']):
    data = tb.loadyaml('./data.yaml', testname)
    datadir = os.environ['SAR_META_DATA_PATH'] if 'SAR_META_DATA_PATH' in os.environ.keys() else data['rootdir']
    H, W = data['SceneSize'][0:2]
    Na, Nr = data['EchoSize'][0:2]
    nspertask = data['nspertask']
    testfile = datadir + '%s/%s/%sMetaSARTasks%dPoints%dSceneSize%dx%dEchoSize%dx%d.h5' % \
                        (data['degree'], data['scatter'], data['degree'], nstest, nspertask, H, W, Na, Nr)

    keys = ['Sr', 'SiFISTA', 'X']
    Sr, Si, X = loaddata(testfile, keys=keys, isrmmean=False)
    SrTest = th.cat((SrTest, Sr))
    SiTest = th.cat((SiTest, Si))
    XTest = th.cat((XTest, X))

if sch['nnmode'] == 'mf':
    M, N = data['EchoSize'][0] * data['EchoSize'][1], data['SceneSize'][0] * data['SceneSize'][1]
    sarmodelfile = '../SARModel/SARModel%sSize%dx%d.mat' % (data['sarmodel'], M, N)
    Phi = th.from_numpy(tb.loadmat(sarmodelfile)['Phi'])
    SrTest = raw2img(SrTest, Phi)

devicename = 'E5 2696v3' if device == 'cpu' else th.cuda.get_device_name(int(str(device)[-1]))

cudaTF32, cudnnTF32, benchmark, deterministic = False, False, True, True

print("---Device: ", device)
print("---Device Name: ", devicename)
print("---Torch Version: ", th.__version__)
print("---Torch CUDA Version: ", th.version.cuda)
print("---CUDNN Version: ", th.backends.cudnn.version())
print("---CUDA TF32: ", cudaTF32)
print("---CUDNN TF32: ", cudnnTF32)
print("---CUDNN Benchmark: ", benchmark)
print("---CUDNN Deterministic: ", deterministic)

th.backends.cuda.matmul.allow_tf32 = cudaTF32
th.backends.cudnn.allow_tf32 = cudnnTF32
th.backends.cudnn.benchmark = benchmark
th.backends.cudnn.deterministic = deterministic

lossfn = [eval(x) for x in sch['loss']['funcstr']]
w = sch['loss']['weight']

if iscplx:
    SrTest, SiTest, XTest = SrTest.unsqueeze(1), SiTest.unsqueeze(1), XTest.unsqueeze(1)
else:
    SrTest, SiTest = SrTest.permute(0, 3, 1, 2), SiTest.permute(0, 3, 1, 2)

N = SrTest.shape[0]
nTest = SrTest.shape[0]
print(nTest, N)

checkpoint_path = outfolder + '/weights/'
logdict = th.load(checkpoint_path + sch['maml']['nnfile'], map_location=device)

net = Net(input_channels=inchnl, kernel_size=sch['kernel_size'], channels=sch['channels'], iscomplex=iscplx)
print(net)
net = net.to(device)
print('# of params: ', net.get_params_number())

maml = l2l.algorithms.MAML(net, lr=sch['maml']['lrfast'], first_order=False)
maml.load_state_dict(logdict['network'])

optimizer = eval(sch['maml']['optimizer'])
scheduler = eval(sch['maml']['scheduler']) if type(sch['maml']['scheduler']) is str else sch['maml']['scheduler']
# th.autograd.set_detect_anomaly(True)
# scheduler = None

idxTest = list(range(0, nTest, int(nTest / 20)))

check(maml, SrTest[idxTest], SiTest[idxTest], idxTest, outfolder=outfolder, prefix='test_before', device=device)

ntasksTest = int(nTest / nspertask)
mbs = sch['maml']['bsize']
npoints = nspertask / 2
nmb = int(ntasksTest / mbs)
mbps, mbps2 = int(mbs * npoints), int(mbs * nspertask)  # points number of 1 meta batch
tb.setseed(sch['seed'])

cntTest = 0
adapt_steps = 20
meta_test_error = 0.0
for task in range(ntasksTest):
    # Compute meta-testing loss
    learner = maml.clone()
    batch = [SrTest[cntTest:cntTest + nspertask], SiTest[cntTest:cntTest + nspertask]]
    evaluation_error = fast_adapt(batch, learner, lossfn, w, adapt_steps=adapt_steps, device=device)
    meta_test_error += evaluation_error.item()
    cntTest += nspertask
print('Meta Test Error', meta_test_error / ntasksTest)

check(learner, SrTest[idxTest], SiTest[idxTest], idxTest, outfolder=outfolder, prefix='test', device=device)
