import torch as th
import torchbox as tb


def loaddata(datafile, keys=['Sr', 'Si', 'X'], isrmmean=True, logf=None):

    data = tb.loadh5(datafile, keys)
    Sr, Si, X = th.from_numpy(data[keys[0]]), th.from_numpy(data[keys[1]]), th.from_numpy(data[keys[2]])
    del data

    if isrmmean:
        Sr = Sr - 15.5
    print('Sr: ', Sr.shape, Sr.dtype, file=logf)
    print('Si: ', Si.shape, Si.dtype, file=logf)
    print('X: ', X.shape, X.dtype, file=logf)

    return Sr, Si, X
