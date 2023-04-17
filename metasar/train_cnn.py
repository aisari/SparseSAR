import os
import torch as th
import torchbox as tb
from dataset import loaddata
from network import Net, raw2img
from solver import train, valid, check


device = 'cuda:1'

schname = 'scheme12'
sch = tb.loadyaml('./scheme.yaml', schname)
inchnl, iscplx = (1, True) if sch['nnvtype'] == 'complex' else (2, False)

outfolder = tb.pathjoin('records', schname, 'cnn')
os.makedirs(outfolder) if not os.path.exists(outfolder) else 0
logfile = outfolder + '/train_valid.log'

if os.path.exists(logfile) and (os.path.getsize(logfile) > 0):
    op = input("--->Do you want to overwrite it? (y/n): ")
    if op in ['n', 'no']:
        exit()
        
logf = open(logfile, 'w')

print(sch, file=logf)

SrTrain, SiTrain, XTrain = th.tensor([]), th.tensor([]), th.tensor([])
for trainname, nstrain in zip(sch['dstrain']['name'], sch['dstrain']['ntasks']):
    data = tb.loadyaml('./data.yaml', trainname)
    datadir = os.environ['SAR_META_DATA_PATH'] if 'SAR_META_DATA_PATH' in os.environ.keys() else data['rootdir']
    H, W = data['SceneSize'][0:2]
    Na, Nr = data['EchoSize'][0:2]
    nspertask = data['nspertask']
    trainfile = datadir + '%s/%s/%sMetaSARTasks%dPoints%dSceneSize%dx%dEchoSize%dx%d.h5' % \
                        (data['degree'], data['scatter'], data['degree'], nstrain, nspertask, H, W, Na, Nr)

    keys = ['Sr', 'SiFISTA', 'X']
    Sr, Si, X = loaddata(trainfile, keys=keys, isrmmean=False, logf=logf)
    SrTrain = th.cat((SrTrain, Sr))
    SiTrain = th.cat((SiTrain, Si))
    XTrain = th.cat((XTrain, X))

SrValid, SiValid, XValid = th.tensor([]), th.tensor([]), th.tensor([])
for validname, nsvalid in zip(sch['dsvalid']['name'], sch['dsvalid']['ntasks']):
    data = tb.loadyaml('./data.yaml', validname)
    datadir = os.environ['SAR_META_DATA_PATH'] if 'SAR_META_DATA_PATH' in os.environ.keys() else data['rootdir']
    H, W = data['SceneSize'][0:2]
    Na, Nr = data['EchoSize'][0:2]
    nspertask = data['nspertask']
    validfile = datadir + '%s/%s/%sMetaSARTasks%dPoints%dSceneSize%dx%dEchoSize%dx%d.h5' % \
                        (data['degree'], data['scatter'], data['degree'], nsvalid, nspertask, H, W, Na, Nr)

    keys = ['Sr', 'SiFISTA', 'X']
    Sr, Si, X = loaddata(validfile, keys=keys, isrmmean=False, logf=logf)
    SrValid = th.cat((SrValid, Sr))
    SiValid = th.cat((SiValid, Si))
    XValid = th.cat((XValid, X))

if sch['nnmode'] == 'mf':
    M, N = data['EchoSize'][0] * data['EchoSize'][1], data['SceneSize'][0] * data['SceneSize'][1]
    sarmodelfile = '../SARModel/SARModel%sSize%dx%d.mat' % (data['sarmodel'], M, N)
    Phi = th.from_numpy(tb.loadmat(sarmodelfile)['Phi'])
    SrTrain = raw2img(SrTrain, Phi)
    SrValid = raw2img(SrValid, Phi)

losslog = tb.LossLog(outfolder, xlabel='Epoch', logdict={'train': [], 'valid': []})

os.makedirs(outfolder + '/images', exist_ok=True)
os.makedirs(outfolder + '/weights', exist_ok=True)

devicename = 'E5 2696v3' if device == 'cpu' else th.cuda.get_device_name(int(str(device)[-1]))

cudaTF32, cudnnTF32, benchmark, deterministic = False, False, True, True

print("---Device: ", device, file=logf)
print("---Device Name: ", devicename, file=logf)
print("---Torch Version: ", th.__version__, file=logf)
print("---Torch CUDA Version: ", th.version.cuda, file=logf)
print("---CUDNN Version: ", th.backends.cudnn.version(), file=logf)
print("---CUDA TF32: ", cudaTF32, file=logf)
print("---CUDNN TF32: ", cudnnTF32, file=logf)
print("---CUDNN Benchmark: ", benchmark, file=logf)
print("---CUDNN Deterministic: ", deterministic, file=logf)

th.backends.cuda.matmul.allow_tf32 = cudaTF32
th.backends.cudnn.allow_tf32 = cudnnTF32
th.backends.cudnn.benchmark = benchmark
th.backends.cudnn.deterministic = deterministic

lossfn = [eval(x) for x in sch['loss']['funcstr']]
w = sch['loss']['weight']

if iscplx:
    SrTrain, SiTrain, XTrain = SrTrain.unsqueeze(1), SiTrain.unsqueeze(1), XTrain.unsqueeze(1)
    SrValid, SiValid, XValid = SrValid.unsqueeze(1), SiValid.unsqueeze(1), XValid.unsqueeze(1)
else:
    SrTrain, SiTrain = SrTrain.permute(0, 3, 1, 2), SiTrain.permute(0, 3, 1, 2)
    SrValid, SiValid = SrValid.permute(0, 3, 1, 2), SiValid.permute(0, 3, 1, 2)

N = SrTrain.shape[0]
Ns = int(N / 3)
SrValid, SiValid, XValid = SrTrain[-Ns:], SiTrain[-Ns:], XTrain[-Ns:]
SrTrain, SiTrain, XTrain = SrTrain[:2 * Ns], SiTrain[:2 * Ns], XTrain[:2 * Ns]

nTrain, nValid, nValid = SrTrain.shape[0], SrValid.shape[0], SrValid.shape[0]
print(nTrain, nValid, nValid, N, file=logf)

net = Net(input_channels=inchnl, kernel_size=sch['kernel_size'], channels=sch['channels'], iscomplex=iscplx)
print(net, file=logf)
net = net.to(device)
print('# of params: ', net.get_params_number(), file=logf)

optimizer = eval(sch['cnn']['optimizer'])
scheduler = eval(sch['cnn']['scheduler']) if type(sch['cnn']['scheduler']) is str else sch['cnn']['scheduler']
# th.autograd.set_detect_anomaly(True)
# scheduler = None

idxTrain = list(range(0, nTrain, int(nTrain / 10)))
idxValid = list(range(0, nValid, int(nValid / 10)))
checkpoint_path = outfolder + '/weights/'
logdict = {}
logf.close()


tb.setseed(sch['seed'])
for epoch in range(sch['nepochs']):
    logf = open(logfile, 'a+')
    lossvtrain = train(net, epoch, SrTrain, SiTrain, XTrain, sch['cnn']['bsize'], lossfn, w=w, optimizer=optimizer, device=device, logf=logf)

    lossvvalid = valid(net, epoch, SrValid, SiValid, XValid, sch['cnn']['bsize'], lossfn, w=w, device=device, logf=logf)

    losslog.add('train', lossvtrain)
    losslog.add('valid', lossvvalid)
    losslog.plot()
    
    logdict['epoch'] = epoch
    logdict['lossestrain'] = losslog.get('train')
    logdict['lossesvalid'] = losslog.get('valid')
    logdict['network'] = net.state_dict()
    logdict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        scheduler.step()
        logdict['scheduler'] = scheduler.state_dict()

    if epoch % sch['scheck'] == 0:
        check(net, SrTrain[idxTrain], SiTrain[idxTrain], idxTrain, outfolder=outfolder, prefix='train', device=device)
        check(net, SrValid[idxValid], SiValid[idxValid], idxValid, outfolder=outfolder, prefix='valid', device=device)

    flag, proof = losslog.judge('valid')
    if flag:
        th.save(logdict, checkpoint_path + proof + '.pth.tar')
 
    logf.close()
