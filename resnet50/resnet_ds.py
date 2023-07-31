import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
import time
import torchvision.models as models
import os
from torchvision import datasets

def add_argument():

    parser = argparse.ArgumentParser(description='ResNet50')

    #data
    parser.add_argument('data', metavar='DIR', nargs='?', 
                        default='../../../datasets/imagenet/images',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('--dummy', 
                        action='store_true', 
                        help="use fake data to benchmark")
    

    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=256,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=14,
                        type=int,
                        help='number of total epochs (default: 14)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")

    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')

    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    parser.add_argument('--num-experts',
                        default=1,
                        type=int,
                        help='(moe) number of total experts')
    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    parser.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )
    parser.add_argument(
        '--noisy-gate-policy',
        default=None,
        type=str,
        help=
        '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
    )
    parser.add_argument(
        '--moe-param-group',
        default=False,
        action='store_true',
        help=
        '(moe) create separate moe param groups, required when using ZeRO w. MoE'
    )
    parser.add_argument('--steps', default=200, type=int,
                    metavar='N', help='number of iterations to measure throughput, -1 for disable')


    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


deepspeed.init_distributed()

import torch.nn as nn
import torch.nn.functional as F

args = add_argument()

transform=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

# Data loading code
if args.dummy:
    print("=> Dummy data is used!")
    train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
    val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
else:
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    testset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=1)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=1000,
                                         shuffle=False,
                                         num_workers=2)

if args.moe:
    deepspeed.utils.groups.initialize(ep_size=args.ep_world_size)

net = models.resnet50()

# not used in this example
def create_moe_param_groups(model):
    from deepspeed.moe.utils import is_moe_param

    params_with_weight_decay = {'params': [], 'name': 'weight_decay_params'}
    moe_params_with_weight_decay = {
        'params': [],
        'moe': True,
        'name': 'weight_decay_moe_params'
    }

    for module_ in model.modules():
        moe_params_with_weight_decay['params'].extend([
            p for n, p in list(module_._parameters.items())
            if p is not None and is_moe_param(p)
        ])
        params_with_weight_decay['params'].extend([
            p for n, p in list(module_._parameters.items())
            if p is not None and not is_moe_param(p)
        ])

    return params_with_weight_decay, moe_params_with_weight_decay

parameters = filter(lambda p: p.requires_grad, net.parameters())
if args.moe_param_group:
    parameters = create_moe_param_groups(net)

# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=trainset)

# fp16
fp16 = model_engine.fp16_enabled()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

t0 = time.time()
for epoch in range(1, args.epochs + 1):

    running_loss = 0.0
    print(f"Total steps per epoch: {len(trainloader)}")
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)
        if fp16:
            inputs = inputs.half()
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()

        # print statistics
        running_loss += loss.item()
        if i % args.log_interval == (
                args.log_interval -
                1):  # print every log_interval mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / args.log_interval))
            running_loss = 0.0

        if i >= args.steps and args.steps > 0:
            print('Throughput: {} images/s'.format(args.batch_size * args.steps / (time.time() - t0)), 'Time used: {} s'.format(time.time() - t0))
            exit()

print('Finished Training on batch size: %d, epochs: %d' %
      (args.batch_size, args.epochs))
print(f'Training time: {time.time() - t0}')

#Test the network on the test data

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(
            model_engine.local_rank)).sum().item()

print(f'Accuracy of the network on the test set: {100 * correct / total}%')