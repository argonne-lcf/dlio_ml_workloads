# The MIT License (MIT)
#
# Copyright (c) 2020-2022 NVIDIA CORPORATION. All rights reserved.
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

import re
import numpy as np
import torch
import torch.optim as optim
from .schedulers import MultiStepLRWarmup, CosineAnnealingLRWarmup

try:
    import apex.optimizers as aoptim
    import apex.contrib.optimizers as acoptim
    have_apex = True
except ImportError:
    print("NVIDIA APEX not found")
    have_apex = False

    
def get_lr_schedule(start_lr, scheduler_arg, optimizer, device, logger, last_step = -1):
    #add the initial_lr to the optimizer
    for pgroup in optimizer.param_groups:
        pgroup["initial_lr"] = start_lr
        
    # after-scheduler
    scheduler = None
    scheduler_after = None
    
    #now check
    if scheduler_arg["type"] == "multistep":
        # set the parameters
        milestones = [ int(x) for x in scheduler_arg["milestones"].split(":") ]
        gamma = float(scheduler_arg["decay_rate"])

        # create scheduler: always select this one
        if scheduler_arg["lr_warmup_steps"] >= 0:
            scheduler = MultiStepLRWarmup(optimizer,
                                          warmup_steps = scheduler_arg["lr_warmup_steps"],
                                          warmup_factor = scheduler_arg["lr_warmup_factor"],
                                          milestones = milestones,
                                          gamma = gamma,
                                          last_epoch = last_step,
                                          device = device)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma = gamma, last_epoch = last_step)

        # save back the parameters for better logging
        scheduler_arg["milestones"] = milestones
        scheduler_arg["decay_rate"] = gamma
    
    elif scheduler_arg["type"] == "cosine_annealing":
        # set parameters
        t_max = int(scheduler_arg["t_max"])
        eta_min = 0. if "eta_min" not in scheduler_arg else float(scheduler_arg["eta_min"])

        # create scheduler: always select this one
        if scheduler_arg["lr_warmup_steps"] >= 0:
            scheduler = CosineAnnealingLRWarmup(optimizer,
                                                warmup_steps = scheduler_arg["lr_warmup_steps"],
                                                warmup_factor = scheduler_arg["lr_warmup_factor"],
                                                T_max = t_max,
                                                eta_min = eta_min,
                                                last_epoch = last_step,
                                                device = device)
        else:
            scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = t_max, eta_min = eta_min)

        # save back the parameters for better logging
        scheduler_arg["t_max"] = t_max
        scheduler_arg["eta_min"] = eta_min
    
    else:
        raise ValueError("Error, scheduler type {} not supported.".format(scheduler_arg["type"]))

    # scheduler data
    for key in scheduler_arg:
        logger.log_event(key = "scheduler_" + key, value = scheduler_arg[key])   

    return scheduler


                     
def get_optimizer(pargs, net, logger, comm_size=1, comm_rank=0):
    # these should be constant
    defaults = {"adam_eps": 1e-6}
    
    optimizer = None
    if pargs.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr = pargs.start_lr, eps = defaults["adam_eps"], weight_decay = pargs.weight_decay)
        log_name = "Adam"
    elif pargs.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr = pargs.start_lr, eps = defaults["adam_eps"], weight_decay = pargs.weight_decay)
        log_name = "AdamW"
    elif pargs.optimizer == "LAMB":
        if have_apex:
            optimizer = aoptim.FusedLAMB(net.parameters(), lr = pargs.start_lr,
                                         eps = defaults["adam_eps"],
                                         weight_decay = pargs.weight_decay,
                                         set_grad_none = not pargs.enable_graph)
            log_name = "LAMB"
        else:
            raise NotImplementedError("Error, optimizer LAMB requires APEX")
    elif pargs.optimizer == "MixedPrecisionLAMB":
        if have_apex:
            optimizer = aoptim.FusedMixedPrecisionLamb(net.parameters(), lr = pargs.start_lr,
                                                       step = 0,
                                                       eps = defaults["adam_eps"],
                                                       weight_decay = pargs.weight_decay)
            log_name = "LAMB"
        else:
            raise NotImplementedError("Error, optimizer MixedPrecisionLAMB requires APEX")
    elif pargs.optimizer == "DistributedLAMB":
        from apex.contrib.optimizers.distributed_fused_lamb import DistributedFusedLAMB
        optimizer = DistributedFusedLAMB(net.parameters(),
                                         lr = pargs.start_lr,
                                         eps = defaults["adam_eps"],
                                         weight_decay = pargs.weight_decay,
                                         max_grad_norm = 1.,
                                         step_supports_amp_scaling = True,
                                         overlap_reductions = False,
                                         full_ar = False,
                                         dwu_num_blocks = 1, dwu_num_chunks = 1,
                                         clip_after_ar = True, # WARNING, THAT HAS TO BE TRUE! Set to FALSE just for debug
                                         verbose = False)
        optimizer.set_global_scale(65536.)
        log_name = "LAMB"
    else:
        raise NotImplementedError("Error, optimizer {} not supported".format(pargs.optimizer))

    # log the optimizer parameters
    logger.log_event(key = "opt_name", value = log_name)
    paramgroup = optimizer.param_groups[0]
    for key in [x for x in paramgroup if x != "params"]:
        if isinstance(paramgroup[key], torch.Tensor):
            val = paramgroup[key].item()
        else:
            val = paramgroup[key]
        logger.log_event(key = "opt_" + key, value = val)
    
    return optimizer
