import time
import torch as th
import torchbox as tb


def fast_adapt(batch, learner, lossfn, w=None, adapt_steps=2, device='cpu'):
    data, labels = batch
    N = data.shape[0]
    adapt_ind, eval_ind = slice(0, N, 2), slice(1, N, 2)
    # Separate data into adaptation/evalutation sets
    adapt_data, adapt_labels = data[adapt_ind].to(device), labels[adapt_ind].to(device)
    eval_data, eval_labels = data[eval_ind].to(device), labels[eval_ind].to(device)

    w = [1.] * len(lossfn) if w is None else w
    # Adapt the model
    for step in range(adapt_steps):
        pred = learner(adapt_data)
        # print(pred.shape, adapt_data.shape, adapt_labels.shape)
        train_error = 0.
        for lossfni, wi in zip(lossfn, w):
            if type(lossfni) in [tb.LogSparseLoss, tb.EntropyLoss]:
                train_error += wi * lossfni(pred)
            else:
                train_error += wi * lossfni(pred, adapt_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    pred = learner(eval_data)

    valid_error = 0.
    for lossfni, wi in zip(lossfn, w):
        if type(lossfni) in [tb.LogSparseLoss, tb.EntropyLoss]:
            valid_error += wi * lossfni(pred)
        else:
            valid_error += wi * lossfni(pred, eval_labels)

    return valid_error


def train(net, epoch, Sr, Si, X, bs, lossfn, w=None, optimizer=None, device='cpu', logf=None):
    N = Sr.shape[0]
    nb = int(N / bs)
    net.train()
    idx = tb.randperm(0, N, N)
    Sr, Si, X = Sr[idx], Si[idx], X[idx]
    w = [1.] * len(lossfn) if w is None else w

    lossv = 0.
    tstart = time.time()
    for b in range(nb):
        s, e = b * bs, (b + 1) * bs
        sri, sii, xi = Sr[s:e], Si[s:e], X[s:e]
        sri, sii, xi = sri.to(device), sii.to(device), xi.to(device)

        optimizer.zero_grad()
        psii = net(sri)

        loss = 0.
        for lossfni, wi in zip(lossfn, w):
            if type(lossfni) in [tb.LogSparseLoss, tb.EntropyLoss]:
                loss += wi * lossfni(psii)
            else:
                loss += wi * lossfni(psii, sii)
        lossv += loss.item()

        loss.backward()

        optimizer.step()

    lossv /= nb
    tend = time.time()

    print('--->Train Epoch: %d, loss: %.4f, time: %.4fs' % (epoch, lossv, tend - tstart), file=logf)
    return lossv


def valid(net, epoch, Sr, Si, X, bs, lossfn, w=None, device='cpu', logf=None):

    net.eval()
    N = Sr.shape[0]
    nb = int(N / bs)
    w = [1.] * len(lossfn) if w is None else w

    with th.no_grad():
        lossv = 0.
        tstart = time.time()
        for b in range(nb):
            s, e = b * bs, (b + 1) * bs
            sri, sii, xi = Sr[s:e], Si[s:e], X[s:e]
            sri, sii, xi = sri.to(device), sii.to(device), xi.to(device)

            psii = net(sri)

            loss = 0.
            for lossfni, wi in zip(lossfn, w):
                if type(lossfni) in [tb.LogSparseLoss, tb.EntropyLoss]:
                    loss += wi * lossfni(psii)
                else:
                    loss += wi * lossfni(psii, sii)
            lossv += loss.item()

        lossv /= nb
        tend = time.time()
        print('--->Valid Epoch: %d, loss: %.4f, time: %.4fs' % (epoch, lossv, tend - tstart), file=logf)
        return lossv


def test(net, epoch, Sr, Si, X, bs, lossfn, w=None, device='cpu', logf=None):

    net.eval()
    N = Sr.shape[0]
    nb = int(N / bs)
    w = [1.] * len(lossfn) if w is None else w

    with th.no_grad():
        lossv = 0.
        tstart = time.time()
        for b in range(nb):
            s, e = b * bs, (b + 1) * bs
            sri, sii, xi = Sr[s:e], Si[s:e], X[s:e]
            sri, sii, xi = sri.to(device), sii.to(device), xi.to(device)

            psii = net(sri)

            loss = 0.
            for lossfni, wi in zip(lossfn, w):
                if type(lossfni) in [tb.LogSparseLoss, tb.EntropyLoss]:
                    loss += wi * lossfni(psii)
                else:
                    loss += wi * lossfni(psii, sii)
            lossv += loss.item()

        lossv /= nb
        tend = time.time()
        print('--->Test Epoch: %d, loss: %.4f, time: %.4fs' % (epoch, lossv, tend - tstart), file=logf)
        return lossv


def check(net, sr, si, idx, outfolder='snapshot/', prefix='train', device='cpu'):

    net.eval()
    with th.no_grad():
        psi = net(sr.to(device))
    saveimg(psi, sr, si, idx, outfolder=outfolder + '/images/', prefix=prefix)


def saveimg(psi, sr, si, idx, outfolder='./snapshot/images/', prefix='train'):

    if si.shape[1] == 2:
        psi = psi.permute(0, 2, 3, 1)
        sr = sr.permute(0, 2, 3, 1)
        si = si.permute(0, 2, 3, 1)
    psi = psi.pow(2).sum(-1).sqrt()
    sr = sr.pow(2).sum(-1).sqrt()
    si = si.pow(2).sum(-1).sqrt()
    psi = psi.squeeze(1)
    sr = sr.squeeze(1)
    si = si.squeeze(1)

    psi, sr, si = tb.mapping(psi), tb.mapping(sr), tb.mapping(si)
    psi = tb.gray2rgb(psi, 'parula', drange=(0, 255), fmtstr=th.uint8)
    sr = tb.gray2rgb(sr, 'parula', drange=(0, 255), fmtstr=th.uint8)
    si = tb.gray2rgb(si, 'parula', drange=(0, 255), fmtstr=th.uint8)
    psi = psi.cpu().detach().numpy()
    sr = sr.cpu().detach().numpy()
    si = si.cpu().detach().numpy()
    for i, ii in zip(range(len(idx)), idx):
        outfileX0 = outfolder + prefix + '_original' + str(ii) + '.tif'
        outfileX = outfolder + prefix + '_reconstruct' + str(ii) + '.tif'
        outfileY = outfolder + prefix + '_groundtruth' + str(ii) + '.tif'
        tb.imsave(outfileX0, sr[i])
        tb.imsave(outfileX, psi[i])
        tb.imsave(outfileY, si[i])
