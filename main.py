"""Train CIFAR10 with PyTorch."""
from __future__ import print_function
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import time
from models import *
from torch.optim import Adam, SGD, RMSprop
from optimizers import *


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--total_epoch', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--decay_epoch', default=150, type=int, help='Number of epochs to decay learning rate')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet', 'vgg', 'vgg_bn'])
    parser.add_argument('--optim', default='adam', type=str, help='optimizer',
                        choices=['sgd', 'adam', 'adamw', 'adabelief', 'yogi', 'msvag', 'radam', 'fromage', 'adabound','theopoula',
                                  ,'exptamed_v0',
                                 'amsgrad', 'rmsprop', 'tuslac', 'adamp', 'swats'])
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='learning rate')
    parser.add_argument('--eps_gamma', default=1, type=float)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')

    parser.add_argument('--eps', default=1e-8, type=float, help='eps for var adam')
    parser.add_argument('--eta', default=0, type=float)
    parser.add_argument('--r', default=0, type=float)

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta', default=1e12, type=float)
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--reset', action = 'store_true',
                        help='whether reset optimizer at learning rate decay')
    return parser


def build_dataset(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                                               num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True,
                                                   num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)


    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader


def get_ckpt_name(dataset='cifar10', seed=111, model='resnet', optimizer='sgd', lr=0.1, final_lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, gamma=1e-3, eps=1e-8, weight_decay=5e-4,
                  reset = False, run = 0, weight_decouple = False, rectify = False, lr_gamma=0.1, eps_gamma=0.1, beta=1e10):
    name = {
        'sgd': 'seed{}-lr{}-momentum{}-wdecay{}-run{}'.format(seed, lr, momentum,weight_decay, run),
        'adam': 'seed{}-lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(seed, lr, beta1, beta2,weight_decay, eps, run),
        'swats': 'seed{}-lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(seed, lr, beta1, beta2, weight_decay, eps, run),
        'amsgrad': 'seed{}-lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(seed, lr, beta1, beta2, weight_decay, eps, run),
        'rmsprop': 'seed{}-lr{}-wdecay{}-run{}'.format(seed, lr, weight_decay, run),
        'theopoula': 'seed{}-lr{}-eps{}-wdecay{}-run{}-lrgamma{}-epsgamma{}'.format(seed, lr, eps, weight_decay, run, lr_gamma, eps_gamma) + '-beta %.1e'%(beta),
        'exptamed_v0': 'seed{}-lr{}-eps{}-wdecay{}-run{}-lrgamma{}-epsgamma{}'.format(seed, lr, eps, weight_decay, run, lr_gamma, eps_gamma) + '-beta %.1e'%(beta),
        'tuslac': 'seed{}-lr{}-wdecay{}-run{}-lrgamma{}-epsgamma{}'.format(seed, lr, weight_decay, run, lr_gamma, eps_gamma) + '-beta %.1e' % (beta),
        'fromage': 'seed{}-lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(seed, lr, beta1, beta2,weight_decay, eps, run),
        'radam': 'seed{}-lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(seed, lr, beta1, beta2,weight_decay, eps, run),
        'adamw': 'seed{}-lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(seed, lr, beta1, beta2,weight_decay, eps, run),
        'adamp': 'seed{}-lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(seed, lr, beta1, beta2, weight_decay, eps, run),
        'adabelief': 'seed{}-lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(seed, lr, beta1, beta2, eps, weight_decay, run),
        'adabound': 'seed{}-lr{}-betas{}-{}-final_lr{}-gamma{}-wdecay{}-run{}'.format(seed, lr, beta1, beta2, final_lr, gamma,weight_decay, run),
        'yogi':'seed{}-lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(seed, lr, beta1, beta2, eps,weight_decay, run),
        'msvag': 'seed{}-lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(seed, lr, beta1, beta2, eps,
                                                                    weight_decay, run),
    }[optimizer]
    return '{}-{}-{}-{}-reset{}'.format(dataset, model, optimizer, name, str(reset))


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    net = {
        'resnet': ResNet34,
        'densenet': DenseNet121,
        'vgg': vgg11,
        'vgg_bn': vgg11_bn
    }[args.model](num_classes)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.eta > 0:
      args.eta = np.sqrt(args.lr) * 5e-4
      print(args.eta)
    if args.optim == 'sgd':
        print('lr: %.4f momentum: %.4f weight_decay: %.4f'%(args.lr, args.momentum, args.weight_decay))
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        print('lr: %.4f betas: %.4f %.4f weight_decay: %.1e eps: %.1e' % (args.lr, args.beta1, args.beta2, args.weight_decay, args.eps))
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'swats':
        print('lr: %.4f betas: %.4f %.4f weight_decay: %.1e eps: %.1e' % (args.lr, args.beta1, args.beta2, args.weight_decay, args.eps))
        return SWATS(model_params, args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim =='amsgrad':
        print('lr: %.4f betas: %.4f %.4f weight_decay: %.1e eps: %.1e' % (
        args.lr, args.beta1, args.beta2, args.weight_decay, args.eps))
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay, amsgrad=True, eps=args.eps)
    elif args.optim == 'rmsprop':
        print('lr: %.4f weight_decay: %.1e' % (args.lr, args.weight_decay))
        return RMSprop(model_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'fromage':
        return Fromage(model_params, args.lr)
    elif args.optim == 'radam':
        return RAdam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adamw':
        return AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adamp':
        return AdamP(model_params, args.lr, betas=(args.beta1, args.beta2),
                     weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adabelief':
        return AdaBelief(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'yogi':
        return Yogi(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'msvag':
        return MSVAG(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'theopoula':
        print('lr: %.4f weight_decay: %.1e eps: %.1e beta: %.1e' % (args.lr, args.weight_decay, args.eps, args.beta))
        return THEOPOULA(model_params, args.lr, eps=args.eps, weight_decay=args.weight_decay, beta=args.beta, eta=args.eta, r=args.r)
    elif args.optim == 'exptamed_v0':
        print('lr: %.4f weight_decay: %.1e eps: %.1e beta: %.1e' % (args.lr, args.weight_decay, args.eps, args.beta))
        return EXPTAMED_v0(model_params, args.lr, eps=args.eps, weight_decay=args.weight_decay, beta=args.beta, eta=args.eta, r=args.r)
    
    elif args.optim == 'tuslac':
        print('lr: %.4f weight_decay: %.1e beta: %.1e' % (args.lr, args.weight_decay, args.beta))
        return TUSLAc(model_params, args.lr, weight_decay=args.weight_decay, beta=args.beta)
    elif args.optim == 'adabound':
        print('optimizer: {}'.format(args.optim) + 'lr: %.4f betas: %.4f %.4f weight_decay: %.1e eps: %.1e final_lr: %.3f gamma: %.3f' % (
            args.lr, args.beta1, args.beta2, args.weight_decay, args.eps, args.final_lr, args.gamma))
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)

    else:
        print('Optimizer not found')

def train(net, epoch, device, data_loader, optimizer, criterion, args):
    print('\nEpoch: %d' % epoch)
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)

    return accuracy


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(' test acc %.3f' % accuracy)

    return accuracy

def adjust_learning_rate(optimizer, epoch, args, step_size=150, gamma=0.1, eps_gamma=1, reset = False):

    for param_group in optimizer.param_groups:
        if epoch % step_size==0 and epoch>0:
            param_group['lr'] *= gamma
            if args.optim == 'theopoula':
                param_group['eps'] *= eps_gamma

    if  epoch % step_size==0 and epoch>0 and reset:
        optimizer.reset()

def main():
    parser = get_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        print("WARNING: You have a CUDA device, use it if you do not yet")
    
        

    

    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    ckpt_name = get_ckpt_name(dataset=args.dataset, seed=args.seed, model=args.model, optimizer=args.optim, lr=args.lr,
                              final_lr=args.final_lr, momentum=args.momentum,
                              beta1=args.beta1, beta2=args.beta2, gamma=args.gamma,
                              eps = args.eps,
                              reset=args.reset, run=args.run,
                              weight_decay = args.weight_decay, lr_gamma=args.lr_gamma, eps_gamma=args.eps_gamma, beta=args.beta)
    print('ckpt_name')
    print("parameters are printed now ")
    par_string=f"lr={args.lr}, final_lr={args.final_lr},eps = {args.eps},weight_decay = {args.weight_decay},lr_gamma={args.lr_gamma}, beta={args.beta}"
    print(par_string)
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']

        curve = os.path.join('curve', ckpt_name)
        curve = torch.load(curve)
        train_accuracies = curve['train_acc']
        test_accuracies = curve['test_acc']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1
        train_accuracies = []
        test_accuracies = []

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    if args.resume:
        optimizer = ckpt['optimizer']
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print('current_lr %.4', current_lr)
    else:
        optimizer = create_optimizer(args, net.parameters())

   # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    

    for epoch in range(start_epoch + 1, args.total_epoch):
        start = time.time()
        #scheduler.step()

        adjust_learning_rate(optimizer, epoch, args, step_size=args.decay_epoch, gamma=args.lr_gamma, eps_gamma=args.eps_gamma, reset=args.reset)
        train_acc = train(net, epoch, device, train_loader, optimizer, criterion, args)
        test_acc = test(net, device, test_loader, criterion)
        end = time.time()
        print('Time: {}'.format(end-start))

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'optimizer': optimizer
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('curve', ckpt_name))


if __name__ == '__main__':
    main()
