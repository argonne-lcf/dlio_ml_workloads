# The MIT License (MIT)
#
# Modifications Copyright (c) 2020-2023 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Basics
import os
import sys
import numpy as np
import datetime as dt
import subprocess as sp

# logging
# wandb
have_wandb = False
try:
    import wandb
    have_wandb = True
except ImportError:
    pass

# mlperf logger
from mlperf_logging import mllog
from mlperf_common.logging import MLLoggerWrapper
from mlperf_common.frameworks.pyt import PyTCommunicationHandler

# Torch
import torch
import torch.optim as optim
from torch.autograd import Variable

# Custom
from driver import Trainer, train_epoch
from driver import Validator, validate
from utils import parser as prs
from utils import losses
from utils import optimizer_helpers as oh
from utils import bnstats as bns
from data import get_dataloaders, get_datashapes
from architecture import deeplab_xception

# DDP
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

# amp
import torch.cuda.amp as amp

#comm wrapper
from utils import comm

# custom logging
from utils.profile_logger import PLogger
from pfw_utils.utility import Profile, PerfTrace
dlp = Profile("train")
#main function
def main(pargs):

    # this should be global
    global have_wandb

    # torch inductor settings
    #torch._inductor.config.triton.cudagraphs = False
    
    #init distributed training
    comm_local_group = comm.init(pargs.wireup_method,
                                 pargs.batchnorm_group_size,
                                 pargs.batchnorm_group_stride)
    comm_rank = comm.get_data_parallel_rank()
    comm_local_rank = comm.get_local_rank()
    comm_size = comm.get_data_parallel_size()
    comm_local_size = comm.get_local_size()
    comm_is_root = (comm_rank == comm.get_data_parallel_root())
    
    # set up logging
    pargs.logging_frequency = max([pargs.logging_frequency, 0])
    if int(os.environ.get("NUM_INSTANCES", "1")) > 1:
        log_file = os.path.normpath(os.path.join(pargs.output_dir, pargs.run_tag + f"_{pargs.experiment_id}.log"))
        mllog.config(filename=log_file)
    #logger = mll.mlperf_logger(log_file, "deepcam",
    #                           "SUBMISSION_ORG_PLACEHOLDER",
    #                           comm_size // comm_local_size)
    logger = MLLoggerWrapper(PyTCommunicationHandler(), value=None)
    PerfTrace.initialize_log(pargs.output_dir, os.path.abspath(pargs.data_dir_prefix))

    # start the logging
    logger.start(key=mllog.constants.INIT_START, sync=True)
    logger.event(key=mllog.constants.CACHE_CLEAR, value=True)

    # submission log
    logger.mlperf_submission_log(benchmark="deepcam", num_nodes=(comm_size // comm_local_size))
    
    #set seed
    seed = pargs.seed
    logger.event(key=mllog.constants.SEED, value=seed)
    
    # Some setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda", comm_local_rank)
        torch.cuda.manual_seed(seed)
        #necessary for AMP to work
        torch.cuda.set_device(device)
        if pargs.deterministic:
            print("WARNING, running in deterministic mode!")
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = not pargs.disable_tuning
    else:
        device = torch.device("cpu")

    # set up profile logger
    log_file = os.path.normpath(os.path.join(pargs.output_dir, pargs.run_tag + f"_{pargs.experiment_id}_rank{comm_rank}.plog"))
    plog = PLogger.getInstance(device=device, logfile=log_file, enabled=pargs.enable_plog, enable_nvml_logging=pargs.enable_nvml_logging)
    plog.event(plog.INTERVAL_START, key="init_start")
        
    #set up directories
    root_dir = os.path.join(pargs.data_dir_prefix)
    output_dir = pargs.output_dir
    plot_dir = os.path.join(output_dir, "plots")
    if comm_is_root:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    
    # Setup WandB
    if not pargs.enable_wandb:
        have_wandb = False
    if have_wandb and comm_is_root:
        # get wandb api token
        certfile = os.path.join(pargs.wandb_certdir, ".wandbirc")
        try:
            with open(certfile) as f:
                token = f.readlines()[0].replace("\n","").split()
                wblogin = token[0]
                wbtoken = token[1]
        except IOError:
            print("Error, cannot open WandB certificate {}.".format(certfile))
            have_wandb = False

        if have_wandb:
            # log in: that call can be blocking, it should be quick
            sp.call(["wandb", "login", wbtoken])
        
            #init db and get config
            resume_flag = pargs.run_tag if pargs.resume_logging else False
            wandb.init(entity = wblogin, project = 'deepcam', 
                       dir = output_dir,
                       name = pargs.run_tag, id = pargs.run_tag, 
                       resume = resume_flag)
            config = wandb.config

    # Logging hyperparameters
    # concurrency logging
    logger.event(key="number_of_ranks", value=comm_size)
    logger.event(key="number_of_nodes", value=(comm_size // comm_local_size))
    logger.event(key="accelerators_per_node", value=comm_local_size)
    
    # basic logging
    logger.event(key="checkpoint", value=pargs.checkpoint)
    logger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=(pargs.local_batch_size * comm_size))
    logger.event(key="batchnorm_group_size", value = pargs.batchnorm_group_size)
    logger.event(key="batchnorm_group_stride", value = pargs.batchnorm_group_stride)
    logger.event(key="gradient_accumulation_frequency", value = pargs.gradient_accumulation_frequency)
    # data option logging
    logger.event(key="data_format", value = pargs.data_format)
    logger.event(key="shuffle_mode", value = pargs.shuffle_mode)
    logger.event(key="data_oversampling_factor", value = pargs.data_oversampling_factor)
    logger.event(key="synchronous_staging", value = pargs.synchronous_staging)
    # perf option logging
    logger.event(key="precision_mode", value = pargs.precision_mode)
    logger.event(key="enable_nhwc", value = pargs.enable_nhwc)
    logger.event(key="enable_graph", value = pargs.enable_graph)
    logger.event(key="enable_jit", value = pargs.enable_jit)
    logger.event(key="enable_gds", value = pargs.enable_gds)
    logger.event(key="enable_mmap", value = pargs.enable_mmap)
    logger.event(key="enable_odirect", value = pargs.enable_odirect)
    logger.event(key="disable_comm_overlap", value = pargs.disable_comm_overlap)
    
    # sanity checks
    assert(pargs.gradient_accumulation_frequency == 1), "Error, gradient_accumulation_frequency != 1 not supported."
    
    # Define architecture
    n_input_channels = len(pargs.channels)
    n_output_channels = 3
    net = deeplab_xception.DeepLabv3_plus(n_input = n_input_channels, 
                                          n_classes = n_output_channels, 
                                          os=16, pretrained=False, 
                                          rank = comm_rank,
                                          process_group = comm_local_group,
                                          enable_gbn = pargs.enable_groupbn)
    net.to(device)
    
    # convert model to NHWC
    if pargs.enable_nhwc:
        net = net.to(memory_format = torch.channels_last)

    if pargs.precision_mode == "fp16":
        net = net.half()

    # get stats handler here
    inplace = True
    if ((comm_local_group is not None) and (comm_local_group.size() > 1)) or pargs.enable_graph:
        inplace = False
    bnstats_handler = bns.BatchNormStatsSynchronize(net, reduction = "mean", inplace = inplace)
        
    #some magic numbers
    loss_pow = -0.125
    class_weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
    fpw_1 = 2.61461122397522257612
    fpw_2 = 1.71641974795896018744
    # loss selection
    criterion = losses.CELoss(class_weights).to(device)
    # convert criterion to NHWC
    if pargs.enable_nhwc:
        criterion = criterion.to(memory_format = torch.channels_last)
    # convert to half if requested
    if pargs.precision_mode == "fp16":
        criterion = criterion.half()

    #select optimizer
    optimizer = oh.get_optimizer(pargs, net, logger, comm_size = comm_size, comm_rank = comm_rank)
    
    # gradient scaler
    gscaler = amp.GradScaler(enabled = ((pargs.precision_mode == "amp") or (pargs.precision_mode == "fp16")))
    
    #restart from checkpoint if desired
    if pargs.checkpoint is not None:
        checkpoint = torch.load(pargs.checkpoint, map_location = device)
        start_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['model'])
    else:
        start_step = 0
        start_epoch = 0
    
    #broadcast model and optimizer state
    steptens = torch.tensor(np.array([start_step, start_epoch]), requires_grad=False).to(device)
    if dist.is_initialized():
        dist.broadcast(steptens, src=comm.get_data_parallel_root(global_rank=True), group=comm.get_data_parallel_group())

    #unpack the bcasted tensor
    start_step = int(steptens.cpu().numpy()[0])
    start_epoch = int(steptens.cpu().numpy()[1])
    
    #select scheduler
    scheduler = None
    if pargs.lr_schedule:
        pargs.lr_schedule["lr_warmup_steps"] = pargs.lr_warmup_steps
        pargs.lr_schedule["lr_warmup_factor"] = pargs.lr_warmup_factor
        scheduler = oh.get_lr_schedule(pargs.start_lr, pargs.lr_schedule, optimizer,
                                       device, logger, last_step = start_step)

    if (pargs.min_epochs > 0) and comm_is_root:
        print("Minimum number of epochs requested. Registering stop scheduler to avoid overfitting.")
    
    ## print parameters
    if (not pargs.enable_wandb) and comm_is_root:
        print(net)
        print("Number of trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
        
    # get input shapes for the upcoming model preprocessing
    # input_shape:
    dshape, label_shape = get_datashapes()
    input_shape = tuple([dshape[2], dshape[0], dshape[1]])
    
    #distributed model parameters
    bucket_cap_mb = 25
    if (pargs.batchnorm_group_size > 1) or pargs.disable_comm_overlap:
        bucket_cap_mb = 220
    dist._DEFAULT_FIRST_BUCKET_BYTES = bucket_cap_mb * 1024 * 1024
        
    # get stream, relevant for graph capture
    scaffolding_stream = torch.cuda.current_stream() if not pargs.enable_graph else torch.cuda.Stream()

    net_validation = net
    if dist.is_initialized() and (pargs.ddp_mode == "full"):
        with torch.cuda.stream(scaffolding_stream):
            ddp_net = DDP(net, device_ids=[device.index],
                          output_device=device.index,
                          find_unused_parameters=False,
                          broadcast_buffers=False,
                          bucket_cap_mb=bucket_cap_mb,
                          gradient_as_bucket_view=True,
                          static_graph=True,
                          process_group=comm.get_data_parallel_group())
        torch.cuda.current_stream().wait_stream(scaffolding_stream)
    else:
        ddp_net = net
        
    # Set up the data feeder
    if comm_is_root:
        print("Creating Dataloaders", flush=True)
    train_loader, train_size, validation_loader, validation_size = get_dataloaders(pargs, root_dir, device, seed, comm_size, comm_rank)

    # log size of datasets
    logger.event(key=mllog.constants.TRAIN_SAMPLES, value=train_size)
    logger.event(key=mllog.constants.EVAL_SAMPLES, value=validation_size)
    
    # create trainer object
    if comm_is_root:
        print("Creating Trainer", flush=True)
    trainer = Trainer(pargs, ddp_net, criterion, optimizer, gscaler, scheduler, device)
    
    # preprocess trainer
    #preprocess(input_shape, label_shape, scaffolding_stream = scaffolding_stream)
    
    # create validator object
    if comm_is_root:
        print("Creating Validator", flush=True)
    validator = Validator(pargs, net_validation, criterion, device)
    gpool = None if trainer.graph is None else trainer.graph.pool()
    #validator.preprocess(input_shape, label_shape, scaffolding_stream = scaffolding_stream, graph_pool = gpool)

    # potential compilation
    if comm_is_root:
        print("Compiling Trainer", flush=True)
    trainer._compile(input_shape)
    if comm_is_root:
        print("Compiling Validator", flush=True)
    validator._compile(input_shape)

    # warmup
    if comm_is_root:
        print("Warmup Trainer", flush=True)
    trainer._warmup(input_shape, label_shape, scaffolding_stream)
    if comm_is_root:
        print("Warmup Validator", flush=True)
    validator._warmup(input_shape, label_shape, scaffolding_stream)

    # potential graph capture
    if comm_is_root:
        print("Capturing Trainer", flush=True)
    trainer._capture(input_shape, label_shape, graph_stream = scaffolding_stream, num_warmup = 1)
    
    # Train network
    if have_wandb and not pargs.enable_jit and not pargs.enable_graph and comm_is_root:
        wandb.watch(trainer.model)
    
    step = start_step
    epoch = start_epoch
    current_lr = pargs.start_lr if not pargs.lr_schedule else scheduler.get_last_lr()[0]
    run_stop_logged = False

    # if we are prefetching to disk, do it now
    if pargs.data_format in ["dali-es-disk"]:
        # only log if we do sync staging
        if pargs.synchronous_staging:
            plog.event(plog.INTERVAL_START, key=mllog.constants.STAGING_START)
            logger.start(key=mllog.constants.STAGING_START)

        # start staging
        train_loader.start_prefetching()

        # sync here if we do fully sync staging
        if pargs.synchronous_staging:
            train_loader.finalize_prefetching()
            #validation_loader.finalize_prefetching()
            plog.event(plog.INTERVAL_END, key=mllog.constants.STAGING_STOP)
            logger.end(key=mllog.constants.STAGING_STOP, sync=True)
    
    # start trining
    plog.event(plog.INTERVAL_END, key="init_stop")
    logger.log_init_stop_run_start()
    plog.event(plog.INTERVAL_START, key="run_start")

    # start staging
    if pargs.data_format in ["dali-es", "dali-es-fused", "dali-es-gpu", "dali-es-gpu-fused"]:
        # only log if we do sync staging
        if pargs.synchronous_staging:
            plog.event(plog.INTERVAL_START, key=mllog.constants.STAGING_START)
            logger.start(key=mllog.constants.STAGING_START)

        # start staging
        train_loader.start_prefetching()

        # sync here if we do fully sync staging
        if pargs.synchronous_staging:
            train_loader.finalize_prefetching()
            #validation_loader.finalize_prefetching()
            plog.event(plog.INTERVAL_END, key=mllog.constants.STAGING_STOP)
            logger.end(key=mllog.constants.STAGING_STOP, sync=True)
    
    # training loop
    while True:

        # start epoch
        plog.event(plog.INTERVAL_START, key="epoch_start", metadata={'epoch_num': epoch+1, 'step_num': step}) 
        logger.start(key=mllog.constants.EPOCH_START, metadata = {'epoch_num': epoch+1, 'step_num': step}, sync=False)

        if pargs.data_format == "hdf5":
            distributed_train_sampler.set_epoch(epoch)

        # train 
        step = train_epoch(pargs, comm_rank, comm_size,
                           step, epoch, trainer,
                           train_loader,
                           logger, have_wandb,
                           disable_scheduler=run_stop_logged)

        if not pargs.disable_validation:
            ## impute values for gbn
            #if pargs.force_groupbn:
            #    bnstats_handler.impute()
            
            # sync bs stats
            bnstats_handler.synchronize()
                    
            # validation
            target_reached = validate(pargs, comm_rank, comm_size,
                                      step, epoch, validator,
                                      validation_loader, 
                                      logger, have_wandb)

        # log the epoch
        plog.event(plog.INTERVAL_END, key="epoch_stop", metadata={'epoch_num': epoch+1, 'step_num': step})
        logger.end(key=mllog.constants.EPOCH_STOP, metadata = {'epoch_num': epoch+1, 'step_num': step}, sync=False)
        epoch += 1
            
        #save model if desired
        if (pargs.save_frequency > 0) and (epoch % pargs.save_frequency == 0):
            logger.start(key = "save_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync=True)
            if comm_is_root:
                checkpoint = {
                    'step': step,
                    'epoch': epoch,
                    'model': trainer.model.state_dict(),
                    'optimizer': optimizer.state_dict()
		}
                torch.save(checkpoint, os.path.join(output_dir, pargs.model_prefix + "_step_" + str(step) + ".cpt") )
                logger.end(key = "save_stop", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync=True)

        # emit run_stop:
        if target_reached and (not run_stop_logged):
            plog.event(plog.INTERVAL_END, key="run_stop", metadata={'status' : 'success', 'epoch_num': epoch})
            logger.log_run_stop(status=mllog.constants.SUCCESS, epoch=epoch)
            run_stop_logged = True

        # decay LR to zero if we converged
        if target_reached and (pargs.min_epochs > 0):
            trainer.reset_state()
                
        # are we done?
        if (epoch >= pargs.max_epochs) or (target_reached and epoch >= pargs.min_epochs):
            if (not target_reached) and (not run_stop_logged):
                plog.event(plog.INTERVAL_END, key="run_stop", metadata={'status' : 'aborted', 'epoch_num': epoch})
                logger.log_run_stop(status=mllog.constants.ABORTED, epoch=epoch)
            break

    return

if __name__ == "__main__":

    # get parsers
    parser = prs.get_parser()
    parser.add_argument("--synchronous_staging", action='store_true')  
    
    # get arguments
    pargs = parser.parse_args()
    
    #run the stuff
    main(pargs)
