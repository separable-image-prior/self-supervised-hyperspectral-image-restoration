from __future__ import print_function
import os
from skimage import io
import argparse
import numpy
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import util.transforms
from torch.utils.data import Dataset, DataLoader
#import adabound

class ResBlock(nn.Module):
    def __init__(self, nChannels, multiply, att_rate):
        super(ResBlock, self).__init__()
        self.convd1 = nn.Conv2d(nChannels*multiply, nChannels*multiply, kernel_size=3, padding=1, groups=nChannels*multiply)
        self.convp1 = nn.Conv2d(nChannels*multiply, nChannels, kernel_size=1, padding=0)
        self.convd2 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, groups=nChannels)
        self.convp2 = nn.Conv2d(nChannels, nChannels*multiply, kernel_size=1, padding=0)

        self.bn1 = nn.BatchNorm2d(nChannels*multiply)
        self.bn2 = nn.BatchNorm2d(nChannels)
        self.att = AttBlock(nChannels*multiply, att_rate)

    def forward(self, x):
        residual = x

        x = F.relu(self.bn1(x))
        x = self.convp1(self.convd1(x))
        x = F.relu(self.bn2(x))
        x = self.convp2(self.convd2(x))

        #x = x * self.att(x)
        #x += residual

        return x

class AttBlock(nn.Module):
    def __init__(self, in_channels, rate):
        super(AttBlock, self).__init__()
        b_channels = int(in_channels*rate)
        self.l1 = nn.Linear(in_channels, b_channels)
        self.l2 = nn.Linear(b_channels, in_channels)

    def forward(self, x):
        c = x.size()[1]
        x = x.mean(dim=-1).mean(dim=-1)
        x = F.relu(self.l1(x))
        x = F.sigmoid(self.l2(x))
        x = x.view(-1,c,1,1)

        return x

class Zeroshot(nn.Module):
    def __init__(self, nLayers, nChannels, multiply, att_rate):
        super(Zeroshot, self).__init__()
        self.sequential = nn.Sequential()

        self.sequential.add_module('convd_in', nn.Conv2d(nChannels, nChannels*multiply, kernel_size=3, padding=1, groups=nChannels))
        self.sequential.add_module('convp_in', nn.Conv2d(nChannels*multiply, nChannels*multiply, kernel_size=1, padding=0))
        self.sequential.add_module('relu_in', nn.ReLU())
        for i in range(nLayers):
            self.sequential.add_module('resblock_%d'%(i+1), ResBlock(nChannels, multiply, att_rate))
        self.sequential.add_module('convd_out', nn.Conv2d(nChannels*multiply, nChannels*multiply, kernel_size=3, padding=1, groups=nChannels*multiply))
        self.sequential.add_module('convp_out', nn.Conv2d(nChannels*multiply, nChannels, kernel_size=1, padding=0))

    def forward(self, x):
        x = self.sequential(x)
        #x = F.sigmoid(self.sequential(x))

        return x

class TrainDataset(Dataset):
    def __init__(self, noisy_img, crop_size, stride, sigma, minvar, maxvar, transform=None):
        self.noisy_img = noisy_img
        self.transform = transform
        self.h, self.v, self.c = noisy_img.shape
        self.crop_size = crop_size
        self.stride = stride
        self.sigma = sigma
        if minvar<=0.01 and maxvar<=0.01:
            minvar = 0.01
            maxvar = 0.01
        self.minvar = minvar
        self.maxvar = maxvar

    def __len__(self):
        return (self.h-self.crop_size)//self.stride*(self.v-self.crop_size)//self.stride

    def __getitem__(self, idx):
        tx = numpy.random.randint(0, (self.h-self.crop_size)//self.stride+1)*self.stride
        ty = numpy.random.randint(0, (self.v-self.crop_size)//self.stride+1)*self.stride
        t = self.noisy_img[tx:tx+self.crop_size, ty:ty+self.crop_size,:]
        input_img = t+numpy.random.randint(int(max(self.sigma-self.minvar,0)*100), int((self.sigma+self.maxvar)*100), (1,))/100*numpy.random.randn(*t.shape)
        teach_img = t
        sample = {'input_img': input_img, 'teach_img': teach_img}

        if self.transform:
            sample = self.transform(sample)

        return sample

class TestDataset(Dataset):
    def __init__(self, clean_img, noisy_img, transform=None):
        self.clean_img = clean_img
        self.noisy_img = noisy_img
        self.transform = transform
        self.h, self.v, self.c = clean_img.shape

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample = {'input_img': self.noisy_img, 'teach_img': self.clean_img}

        if self.transform:
            sample = self.transform(sample)

        return sample

def train(args, model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, samples in enumerate(loader):
        input_img, teach_img = samples['input_img'].to(device), samples['teach_img'].to(device)
        optimizer.zero_grad()
        output_img = model(input_img)

        loss = torch.sum((teach_img-output_img)**2)/input_img.size()[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc_norm)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input_img), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))
        #oku
    return loss.item()

def test(args, model, device, loader):
    model.eval()
    test_loss=0
    with torch.no_grad():
        samples = next(loader.__iter__())
        input_img, teach_img = samples['input_img'].to(device), samples['teach_img'].to(device)
        output_img = model(input_img)
        test_loss = 10*torch.log10(1./torch.mean((teach_img-output_img)**2))

    print('\nPSNR: {:.4f}\n'.format(test_loss))
    return test_loss

def visualize(args, model, device, loader):
    model.eval()
    test_loss=0
    with torch.no_grad():
        samples = next(loader.__iter__())
        input_img, teach_img = samples['input_img'].to(device), samples['teach_img'].to(device)
        output_img = model(input_img)

        input_img=input_img[0,args.v_band]
        output_img=output_img[0,args.v_band]

        input_img = (input_img-input_img.min())/(input_img.max()-input_img.min())
        input_img[input_img>1]=1
        input_img*= 255
        input_img = input_img.type(torch.uint8).to('cpu').detach().numpy()
        #oku io.imsave('input.png', input_img)

        #output_img = (output_img-output_img.min())/(output_img.max()-output_img.min())
        output_img[output_img>1]=1
        output_img[output_img<0]=0
        output_img*= 255
        output_img = output_img.type(torch.uint8).to('cpu').detach().numpy()
        #oku io.imsave('output.png', output_img)

def est_noise(img):
    r = img.transpose(2,0,1).reshape(img.shape[-1],-1)
    small = 1e-6

    L,N = r.shape
    w = numpy.zeros((L,N))
    RR = numpy.matmul(r,r.T)
    RRi = numpy.linalg.inv(RR+small*numpy.eye(L))
    for i in range(L):
        XX = RRi - numpy.matmul(RRi[:,i].reshape(-1,1),RRi[i,:].reshape(1,-1))/RRi[i,i]
        RRa = RR[:,i]; RRa[i]=0
        beta = numpy.matmul(XX,RRa); beta[i]=0
        w[i,:] = r[i,:] - numpy.matmul(beta.T,r)
    Rw=numpy.diag(numpy.diag(numpy.matmul(w,w.T)/N));
    return w,Rw

#oku
def stepUpdate(scheduler, maxval):
    current_lr = scheduler.get_lr()
    if  current_lr[0] > maxval:
        scheduler.step()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Visual Inspection')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    #oku
    parser.add_argument('--epochs-pre', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1 (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2 (default: 0.999)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--file-path',
                        help='path of image file')
    parser.add_argument('--v-name',
                        help='name of image variable')
    parser.add_argument('--n-layers', type=int, default=16,
                        help='number of conv layers (default: 16)')
    parser.add_argument('--multiply', type=int, default=4,
                        help='output channel of conv layer (default: 4)')
    parser.add_argument('--att-rate', type=float, default=8,
                        help='inverse of bottleneck rate in attention (default: 8)')
    parser.add_argument('--crop-size', type=int, default=40,
                        help='crop size of image when creating training data set (default: 40)')
    parser.add_argument('--stride', type=int, default=10,
                        help='stride width when creating training data set (default: 10)')
    parser.add_argument('--step-size', type=int, default=30,
                        help='the learning rate declines with this step size (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='decay rate of learning rate (default: 0.1)')
    parser.add_argument('--est-noise', action='store_true', default=False,
                        help='enables noise level estimation')
    parser.add_argument('--v-band', type=int, default=25,
                        help='index of band to be visualized (default: 25)')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='variance of noise added for performance evaluation (default: 0.1)')
    parser.add_argument('--minvar', type=float, default=0.05,
                        help='minimum variance of noise added during training (default: 0.05)')
    parser.add_argument('--maxvar', type=float, default=0.05,
                        help='maximum variance of noise added during training (default: 0.05)')
    parser.add_argument('--gc-norm', type=float, default=1.0,
                        help='L2 norm threshold for gradient clipping (default: 1.0)')
    parser.add_argument('--minstep', type=float, default=0.0,
                        help='lower bound for stepsize (default: 0.0)')
    parser.add_argument('--outer-iteration', type=int, default=2,
                        help='number of target switch (default: 1, range:1,2,or3)')
    parser.add_argument('--init-lr1', type=float, default=0.0,
                        help='init lr for second ite. (default: 0.0,  0.0 for no init)')
    parser.add_argument('--init-lr2', type=float, default=0.0,
                        help='init lr for third ite.(default: 0.0,  0.0 for no init)')
    parser.add_argument('--auto-step', type=int, default=0,
                        help='stepsize auto update(default: 0 not excute')
    parser.add_argument('--th-not-update', type=int, default=3,
                        help='# of no-update (default: 2 not excute')
    parser.add_argument('--step-num-hist', type=int, default=5,
                        help='# of history for stepsize update (default: 5')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    train_data_transform = transforms.Compose([
        util.transforms.RotateFlip(),
        util.transforms.ToTensor()])

    test_data_transform = transforms.Compose([
        util.transforms.ToTensor()])


    t = scipy.io.loadmat(args.file_path, squeeze_me=True)[args.v_name]
    source_img = (t-t.min())/(t.max()-t.min())
    target_img = source_img+args.sigma*numpy.random.randn(*source_img.shape)

    if args.est_noise:
        print('Estimating the noise level...')
        w, _ = est_noise(target_img)
        est_level = w.std()
        print('Estimated level: %f'%est_level)

    if args.auto_step == 1:
        step_size = 1
    else:
        step_size = args.step_size

    train_dataset = TrainDataset(
        noisy_img=target_img, crop_size=args.crop_size, stride=args.stride, sigma=est_level if args.est_noise else args.sigma, minvar=args.minvar, maxvar=args.maxvar, transform=train_data_transform)
    test_dataset = TestDataset(
        clean_img=source_img, noisy_img=target_img, transform=test_data_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False, **kwargs)

    model = Zeroshot(nLayers=args.n_layers, nChannels=source_img.shape[-1], multiply=args.multiply, att_rate=1/args.att_rate).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    #optimizer = adabound.AdaBound(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), final_lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args.gamma)

    print('qqq')
    MaxPSNR = 0
    min_loss = 100000000.0
    minLossPSNR = 0
    #oku
    if args.auto_step == 1:
        num_hist = args.step_num_hist
        th_not_update = args.th_not_update

    for out_loop in range(1, args.outer_iteration):
        if args.auto_step == 1:
            loss_hist = numpy.zeros(num_hist)
            loss_ind = 0
            num_not_update = 0
        for epoch in range(1, args.epochs_pre + 1):
            #oku
            #scheduler.step()
            if args.auto_step == 0:
                stepUpdate(scheduler, args.minstep)

            tloss = train(args, model, device, train_loader, optimizer, epoch)

### scheduler update
            print(scheduler.get_lr())
            if args.auto_step == 1:
                ch = numpy.average(loss_hist)
                print(tloss)
                if ch < tloss and loss_ind >= num_hist: num_not_update += 1
                if (loss_ind >= num_hist and  num_not_update >= th_not_update) or  loss_ind > 40:
                    loss_ind = 0
                    num_not_update = 0
                    stepUpdate(scheduler, args.minstep)

                if loss_ind < num_hist:
                    loss_hist[loss_ind] = tloss
                    loss_ind += 1
                else:
                    loss_hist = numpy.roll(loss_hist, -1)
                    loss_hist[-1] = tloss
                    loss_ind += 1
 ### end: scheduler update

            PSNR = test(args, model, device, test_loader)
            if PSNR > MaxPSNR: MaxPSNR = PSNR
            if min_loss > tloss:
                minLossPSNR = PSNR
                min_loss = tloss
            visualize(args, model, device, test_loader)
            print('Train{} Epoch: {}  maxPSNR: {:.4f} minLossPSNR: {:.4f}'.format(out_loop,epoch,MaxPSNR,minLossPSNR))


        samples = next(test_loader.__iter__())
        tt = samples['input_img'].to(device)
        output_img = model(tt)
        output_img=output_img.to('cpu').detach().numpy()
        output_img=numpy.squeeze(output_img)
        output_img=output_img.transpose(1,2,0)
        print(output_img.shape)

        train_dataset = TrainDataset(
            noisy_img=output_img, crop_size=args.crop_size, stride=args.stride, sigma=est_level if args.est_noise else args.sigma, minvar=args.minvar, maxvar=args.maxvar
            , transform=train_data_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        if args.init_lr1 != 0.0:
            print('LR updated 1\n')
            optimizer = optim.Adam(model.parameters(), lr=args.lr*args.init_lr1, betas=(args.beta1, args.beta2))
            #optimizer = adabound.AdaBound(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), final_lr=0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args.gamma)



    min_loss = 100000000.0

    if args.init_lr2 != 0.0:
        print('LR updated 1\n')
        optimizer = optim.Adam(model.parameters(), lr=args.lr*args.init_lr2, betas=(args.beta1, args.beta2))
        #optimizer = adabound.AdaBound(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), final_lr=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args.gamma)

        #oku
    if args.auto_step == 1:
        loss_hist = numpy.zeros(num_hist)
        loss_ind = 0
        num_not_update = 0

    for epoch in range(1, args.epochs + 1):
            #oku
            #scheduler.step()
        if args.auto_step == 0:
            stepUpdate(scheduler, args.minstep)

        tloss = train(args, model, device, train_loader, optimizer, epoch)

### scheduler update
        print(scheduler.get_lr())
        if args.auto_step == 1:
            ch = numpy.average(loss_hist)
            if  ch < tloss and loss_ind >= num_hist: num_not_update += 1
            if (loss_ind >= num_hist and  num_not_update >= th_not_update) or  loss_ind > 40:
                loss_ind = 0
                num_not_update = 0
                stepUpdate(scheduler, args.minstep)

            if loss_ind < num_hist:
                loss_hist[loss_ind] = tloss
                loss_ind += 1
            else:
                loss_hist = numpy.roll(loss_hist, -1)
                loss_hist[-1] = tloss
                loss_ind += 1
 ### end: scheduler update
        PSNR = test(args, model, device, test_loader)
        if PSNR > MaxPSNR: MaxPSNR = PSNR
        if min_loss > tloss:
            minLossPSNR = PSNR
            min_loss = tloss
        visualize(args, model, device, test_loader)
        print('Train final Epoch: {}  maxPSNR: {:.4f} minLossPSNR: {:.4f}'.format(epoch,MaxPSNR,minLossPSNR))

if __name__ == '__main__':
    main()
