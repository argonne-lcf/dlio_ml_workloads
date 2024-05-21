# Adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import os
from enum import Enum
import logging, sys

# this is to have pytorch profiler
from torch.profiler import profile, record_function, ProfilerActivity

import torchvision.models as models
import signal

# 1. Initialize Horovod
import horovod.torch as hvd
hvd.init()

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn

# For DLIO profiler
from pfw_utils.utility import Profile, PerfTrace
dlp = Profile("RESNET50")
#compute_dlp = dlp_event_logging("Compute")
log = logging.getLogger('ResNet50')
log.setLevel(logging.DEBUG)


# def capture_signal(signal_number, frame):
#     log_inst.finalize()
#     print("Kaushik-Calling-Finalize")
#     print('Received Signal {}'.format(signal_number))
#     exit(1)
 
# signal.signal(signal.SIGABRT, capture_signal)
#os.kill(os.getpid(), signal.SIGABRT)
dlp_data = Profile("IO")
class MyImageFolder(datasets.ImageFolder):
#    @io_dlp.log
#    def __getitem__(self, index):
#        return super(MyImageFolder, self).__getitem__(index)
    @dlp_data.log
    def preprocess(self, sample, target):
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
    @dlp_data.log
    def read_data(self, index):
        path, target = self.samples[index]
        return self.loader(path), target
    @dlp_data.log
    def __getitem__(self, index):
        sample, target = self.read_data(index)
        sample, target = self.preprocess(sample, target)
        return sample, target


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

dlp_train = Profile("train")
dlp_eval = Profile("eval")
def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()    

    end = time.time()
    log.info("start training")
    for i, (images, target) in dlp_train.iter(enumerate(train_loader)):
            # measure data loading time
            data_time.update(time.time() - end)
            with Profile(name="H2D", cat="train"):                        
                # move data to the same device as model - cpu-gpu transfer
                images = images.to(device)
                target = target.to(device)
            with Profile(name="compute-forward", cat="train"):
                # compute output
                output = model(images)
                loss = criterion(output, target)
            with Profile(name="compute-backward", cat="train"):
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            if i == 0:
                first_batch_time = time.time() - end
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and hvd.rank() == 0:
                progress.display(i + 1)
            if i == args.steps-1 and args.steps > 0:
                if hvd.rank() == 0:
                    log.info('Throughput: {:.3f} images/s,\n'.format((args.steps-1) * args.batch_size * hvd.size() / (batch_time.sum-first_batch_time))+
                        'Batch size: {}\n'.format(args.batch_size)+
                        'Num of GPUs: {}\n'.format(hvd.size())+
                        'Total time: {:.3f} s\n'.format(batch_time.sum)+
                        'Average batch time: {:.3f} s\n'.format(batch_time.avg)+
                        'First batch time: {:.3f} s\n'.format(first_batch_time))
                return 0
#                log_inst.finalize()



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dlp_eval(test_loader):
            with Profile(name="H2D", cat="eval"):                                    
                data, target = data.to(device), target.to(device)
            with Profile(name="compute", cat="eval"):
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if hvd.rank() == 0: 
        logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('data', metavar='DIR', nargs='?', default='/eagle/datascience/ImageNet/ILSVRC/Data/CLS-LOC/',
                    help='path to dataset (default: imagenet)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--steps', default=10, type=int,
                    metavar='N', help='number of iterations to measure throughput, -1 for disable')
    parser.add_argument('--save_model', default=0, type=int,
                metavar='CK', help='checkpointing, -1 for disable')
    parser.add_argument("--output_folder", default='outputs', type=str)
    parser.add_argument("--profile", action='store_true', help="use pytorch profiler")
    parser.add_argument("--shuffle", action='store_true', help="shuffle the dataset")
    parser.add_argument("--custom_image_loader", action='store_true', help="use custom_image_folder")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dont_pin_memory", action='store_true')
    parser.add_argument("--multiprocessing_context", default=None, type=str)

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    pin_memory = not args.dont_pin_memory
    # create logger with 'spam_application'
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f"{args.output_folder}/resnet50.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.ERROR)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    
    log.info("Horovod: I am worker %s of %s." %(hvd.rank(), hvd.size()))
    
    pfwlogger = PerfTrace.initialize_log(args.output_folder+f"/trace-{hvd.rank()}-of-{hvd.size()}.pfw", os.path.abspath(args.data), process_id = hvd.rank())    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 2. Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())

    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': pin_memory}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)
        
    if args.multiprocessing_context is not None:
        multi = {'multiprocessing_context': args.multiprocessing_context}
        train_kwargs.update(multi)
        val_kwargs.update(multi)

    # Data loading code
    if args.dummy:
        if hvd.rank()==0:
            log.info("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if args.custom_image_loader:
            train_dataset = MyImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            val_dataset = MyImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=args.shuffle)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler= train_sampler, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, sampler=test_sampler, **val_kwargs)

    model = models.resnet50()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # 4. Horovod: scale learning rate by the number of GPUs.
    optimizer = torch.optim.SGD(model.parameters(), args.lr * hvd.size(),
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 5. Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # 6. Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    t0 = time.time()
    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:    
            for epoch in range(1, args.epochs + 1):
                train_sampler.set_epoch(epoch)
                train(train_loader, model, criterion, optimizer, epoch, device, args)
                # test(model, device, val_loader)
                scheduler.step()
        prof.export_chrome_trace(f"{args.output_folder}/trace-{hvd.rank()}.json")
    else:
        for epoch in range(1, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train(train_loader, model, criterion, optimizer, epoch, device, args)
            # test(model, device, val_loader)
            scheduler.step()
        
    if hvd.rank() == 0:
        log.info(time.time()-t0)
        if args.save_model:
            with Profile(name="checkpointing", cat='IO'):
            #with dlp_event_logging("IO", name="checkpointing") as compute:
                torch.save(model.state_dict(), "resnet50.pt")
    pfwlogger.finalize()
    #test(model, device, val_loader)
    #log_inst.finalize()


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        log.info("all_reduce")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        # dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        hvd.allreduce(total, name=self.name)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        log.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
