#!/usr/bin/python3
import json
import csv
import glob
import sys
import os
import shutil
import re
import statistics
from math import floor
# import mllogging
from collections import defaultdict
# from mllogging.compliance_checker import mlp_compliance


# Used for pruning stats logs to fit on ELK
# originally calculated to be 10000 based on rough estimate of 1:3 ratio of  {# of stats lines} : {nested_log size} where max nested log size
# should be < 32k bytes in order to post correct on elk. 2000 however ended up working the best in practice
MAX_STATS_LINES = 1500


# This is a list of metrics to calculate from the log. Each metric is specified as a (<start_tag>, <stop_tag>, <name>)
# tuple. The <name> is the display name to use for the corresponding column(s) in the output CSV.
#
# The script will print statistics for each metric defined by subtracting the timestamps of <stop_tag> with the
# immediately preceding <start_tag>, if any, according to the order they appear in the logfile.
#
# If the tags are identical, then it will print statistics for metric defined by subtracting the timestamps of
# successive occurences of <tag>, as they appear in the logfile. Two notes for this case. Note1: total count will be the
# number of occurences of that tag minus one (except for Note2); Note2: the same tag printed twice in the log with the
# same timestamp will count as one occurence.
time_metrics = [
                ("run_start", "run_stop", 'd_score'),
                ('init_start', 'init_stop', 'd_init'),
                ('eval_start', 'eval_stop', 'd_eval')
            ]
submission_metrics = [
        ('init_start', 'd_gpus_per_node'),
        ('submission_platform', 'd_nodes')
]
# These are a list of parameter settings to extract from the log. Each setting to is specified as a (<tag>, <param>, <name>)
# tuple. The <name> is the display name for the column heading in the output CSV.
#
# The <tag> is a string that appears on the logging line before the <param> string. This can be set to '' if you want a simple
# param that will have no associated tag.
param_metrics = [
                    ('eval_accuracy', '\"epoch_num\"', 'd_epochs', 0, '[\'metadata\'][\'epoch_num\']'),
                    ('run_stop', '\"status\"', 's_status', 'fail', '[\'metadata\'][\'status\']'),
                    ('submission_entry', '\'num_nodes\'', 'd_nodes', 0, '[\'value\'][\'nodes\'][\'num_nodes\']'),
                    ('submission_entry', '\'num_accelerators\'', 'd_gpus_per_node', 0, '[\'value\'][\'nodes\'][\'num_accelerators\']'),
                    ('global_batch_size', '\"value\"', 'd_batch_size', 0, '[\'value\']'),
                ]

hpc_common_hyperparam_metrics = [
    ('submission_benchmark', '\"value\"', 's_submission_benchmark', '', '[\'value\']'),
    ('gradient_accumulation_steps', '\"value\"', 'd_gradient_accumulation_steps', 0, '[\'value\']'),
    ('gradient_accumulation_frequency', '\"value\"', 'd_gradient_accumulation_frequency', 0, '[\'value\']'),
]

cosmoflow_hyperparam_metrics = [
    ('global_batch_size', '\"value\"', 'd_global_batch_size', 0, '[\'value\']'),
    ('opt_name', '\"value\"', 's_opt_name', '', '[\'value\']'),
    ('opt_base_learning_rate', '\"value\"', 'd_opt_base_learning_rate', 0, '[\'value\']'),
    ('opt_learning_rate_warmup_epochs', '\"value\"', 'd_opt_learning_rate_warmup_epochs', 0, '[\'value\']'),
    ('opt_learning_rate_warmup_factor', '\"value\"', 'd_opt_learning_rate_warmup_factor', 0, '[\'value\']'),
    ('opt_learning_rate_decay_boundary_epochs', '\"value\"', 'd_opt_learning_rate_decay_boundary_epochs', 0, '[\'value\']'),
    ('opt_learning_rate_decay_factor', '\"value\"', 'd_opt_learning_rate_decay_factor', 0, '[\'value\']'),
    ('dropout', '\"value\"', 'd_dropout', 0, '[\'value\']'),
    ('opt_weight_decay', '\"value\"', 'd_opt_weight_decay', 0, '[\'value\']'),
    ('eval_error', '\"value\"', 'd_eval_error', 0, '[\'value\']'),
] + hpc_common_hyperparam_metrics

deepcam_hyperparam_metrics = [
    ('gradient_accumulation_frequency', '\"value\"', 'd_gradient_accumulation_frequency', 0, '[\'value\']'),
    ('seed', '\"value\"', 'd_seed', 0, '[\'value\']'),
    ('global_batch_size', '\"value\"', 'd_global_batch_size', 0, '[\'value\']'),
    ('number_of_nodes', '\"value\"', 'd_number_of_nodes', 0, '[\'value\']'),
    ('batchnorm_group_size', '\"value\"', 'd_batchnorm_group_size', 0, '[\'value\']'),
    ('opt_name', '\"value\"', 's_opt_name', '', '[\'value\']'),
    ('opt_lr', '\"value\"', 'd_opt_lr', 0, '[\'value\']'),
    ('opt_betas', '\"value\"', 'd_opt_betas', 0, '[\'value\']'),
    ('opt_eps', '\"value\"', 'd_opt_eps', 0, '[\'value\']'),
    ('scheduler_type', '\"value\"', 's_scheduler_type', '', '[\'value\']'),
    ('scheduler_lr_warmup_steps', '\"value\"', 'd_scheduler_lr_warmup_steps', 0, '[\'value\']'),
    ('scheduler_lr_warmup_factor', '\"value\"', 'd_scheduler_lr_warmup_factor', 0, '[\'value\']'),
    ('train_samples', '\"value\"', 'd_train_samples', 0, '[\'value\']'),
    ('eval_samples', '\"value\"', 'd_eval_samples', 0, '[\'value\']'),
    ('eval_accuracy', '\"value\"', 'd_eval_accuracy', 0, '[\'value\']'),
    ('scheduler_t_max', '\"value\"', 'd_scheduler_t_max', 0, '[\'value\']'),
    ('scheduler_eta_min', '\"value\"', 'd_scheduler_eta_min', 0, '[\'value\']'),
    ('opt_bias_correction', '\"value\"', 'd_opt_bias_correction', 0, '[\'value\']'),
    ('opt_grad_averaging', '\"value\"', 'd_opt_grad_averaging', 0, '[\'value\']'),
    ('opt_max_grad_norm', '\"value\"', 'd_opt_max_grad_norm', 0, '[\'value\']'),
    ('scheduler_milestones', '\"value\"', 'd_scheduler_milestones', 0, '[\'value\']'),
    ('scheduler_decay_rate', '\"value\"', 'd_scheduler_decay_rate', 0, '[\'value\']'),
] + hpc_common_hyperparam_metrics

oc20_hyperparam_metrics = [
    ('global_batch_size', '\"value\"', 'd_global_batch_size', 0, '[\'value\']'),
    ('opt_name', '\"value\"', 's_opt_name', '', '[\'value\']'),
    ('opt_base_learning_rate', '\"value\"', 'd_opt_base_learning_rate', 0, '[\'value\']'),
    ('opt_learning_rate_warmup_steps', '\"value\"', 'd_opt_learning_rate_warmup_steps', 0, '[\'value\']'),
    ('opt_learning_rate_warmup_factor', '\"value\"', 'd_opt_learning_rate_warmup_factor', 0, '[\'value\']'),
    ('opt_learning_rate_decay_boundary_steps', '\"value\"', 'd_opt_learning_rate_decay_boundary_steps', 0, '[\'value\']'),
    ('opt_learning_rate_decay_factor', '\"value\"', 'd_opt_learning_rate_decay_factor', 0, '[\'value\']'),
    ('eval_error', '\"value\"', 'd_eval_error', 0, '[\'value\']'),
] + hpc_common_hyperparam_metrics


gpus_per_node_count = { 'DGX1': 8, \
                            'dgx1': 8, \
                            'DGX2': 16, \
                            'dgx2': 16, \
                            'DGX2H': 16, \
                            'dgx2h': 16, \
                            'DGXA100': 8, \
                            'dgxa100': 8 \
                          }

system_names = { 'DGX1' : 'NVIDIA DGX-1', \
                 'dgx1' : 'NVIDIA DGX-1', \
                 'DGX2' : 'NVIDIA DGX-2', \
                 'dgx2' : 'NVIDIA DGX-2', \
                 'DGX2H': 'NVIDIA DGX2-H', \
                 'dgx2h': 'NVIDIA DGX2-H', \
                 'DGXA100': 'NVIDIA DGX A100', \
                 'dgxa100': 'NVIDIA DGX A100' \
               }

hyperparam_metrics_map = {
    'cosmoflow'              : cosmoflow_hyperparam_metrics,
    'deepcam'                : deepcam_hyperparam_metrics,
    'oc20'                   : oc20_hyperparam_metrics,
}

###################################################
# Post Process Helpers
###################################################
def get_hyperparam_metrics(benchmark):
    return hyperparam_metrics_map[benchmark]

def dir_path(string):
    if os.path.ispath(string):
        return string
    else:
        raise NotADirectoryError(string)

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

def extract_mllog(file):
    mllog = []
    file.seek(0)

    def mll_filter(line):
        mll_json = {}
        new_json_match = re.match(r".*:::MLLOG\s+(.+)", line)
        if new_json_match:
            json_string = new_json_match.group(1)
            mll_json = json.loads(json_string)
        return mll_json

    for line in file.readlines():
        ret = mll_filter(line)
        if ret:
            mllog.append(ret)
    return mllog


###################################################
# Parse and Extract Relevant Values
###################################################

def process_submission(mllog, benchmark, config, params = submission_metrics):
    """
    Find parameter count from logfile
    """
    if "_" in config:
        system_type, _ = config.split("_", 1)
    else:
        system_type = config
    if system_type in ['DGX2']:
        if benchmark in ['reinforcement', 'minigo']:
            system_type = 'DGX2'
        else:
            system_type = 'DGX2H'
    submission_settings = defaultdict()
    for string, param in params:
        submission_settings[param] = 0
    for line in mllog:
        for key, param in params:
            if key == line['key']:
                if "init_start" == key:
                    submission_settings[param] += 1
                if "submission_platform" == key:
                    node_count_match = re.match(r'([0-9]*)'+'x.*', line["value"])
                    if node_count_match:
                        submission_settings[param] = int(node_count_match.group(1))
    if submission_settings['d_gpus_per_node'] > 1 and submission_settings['d_nodes'] > 0:
        submission_settings['d_gpus_per_node'] = int(submission_settings['d_gpus_per_node'] / submission_settings['d_nodes'])
    else:
        submission_settings['d_gpus_per_node'] = gpus_per_node_count[system_type]
    submission_settings['s_system_type'] = system_type
    return submission_settings


def process_params(params, mllog):
    """
    Find parameter settings from logfile
    """
    settings = defaultdict(list)
    for _, _, name, default_value, nested in params:
        settings[name] = default_value

    for line in mllog:
        for tag, param, name, default_value, nested in params:
            if tag == line['key']:
                json_string = json.dumps(line)
                try:
                    metadata_json_obj = json.loads(json_string)
                except:
                    print("illformed line:", line)
                    continue
                try:
                    settings[name] = eval("metadata_json_obj"+nested)
                except:
                    continue
    return settings

"""
multiprocess function returns a dict by searching image_classification logs to identify s_process type as horovod or device, s_network as f(fused) on nf(non fused)
looks for lines
#++ KVSTORE=horovod
#++ KVSTORE=device
#NETWORK="resnet-v1b-normconv-fl" for normconv (fused)
#NETWORK="resnet-v1b-fl" for standard (non fused)
for other benchmarks, it returns an empty dictionary
"""
def multiprocess(name, mllog):
    name = name.split('--')[0]
    if 'image_classification' in name:
        multiprocess = read_logfile_multiprocess_image_classification(mllog)
    else:
        multiprocess = {}
    return multiprocess

def read_logfile_multiprocess_image_classification(mllog):
    multiprocess_dict = {}
    for line in mllog:
        if line['key'] == 's_process':
            multiprocess_dict["s_process"] = line['value']
            next;
        if line['key'] == 's_network':
            network = line['value']
            if ( network == "resnet-v1b-fl"):
                multiprocess_dict["s_network"] = "f"
            elif ( network == "resnet-v1b-normconv-fl"):
                multiprocess_dict["s_network"] = "nf"
            elif( network == "resnet-v1b-normconv2-fl"):
                multiprocess_dict["s_network"] = "nf2"
            elif( network == "resnet-v1b-dbar-fl"):
                multiprocess_dict["s_network"] = "dbar"
            elif( network == "resnet-v1b-dbar2-fl"):
                multiprocess_dict["s_network"] = "dbar2"
            elif(network == "resnet-v1b-stats-fl"):
                multiprocess_dict["s_network"] = "stats"
            else:
                multiprocess_dict["s_network"] = network
            next;
        if ( "s_process", "s_network" ) in multiprocess_dict:
            break;
    if ( "s_process" ) not in multiprocess_dict:
        multiprocess_dict["s_process"] = "unknown"
    if ( "s_network" ) not in multiprocess_dict:
        multiprocess_dict["s_network"] = "unknown"
    return multiprocess_dict


def process_job_data(benchmark, file):
    """
    Find slurm job details from logfile
    """
    file.seek(0)
    jobinfo = {'s_node_list': ''}
    nodes = sorted(re.findall(r'^Clearing cache on (.*)', file.read(), re.MULTILINE))
    if nodes:
        jobinfo['s_node_list'] = " ".join(nodes)
    return jobinfo


###################################################
# Parse and Generate Lines with Timestamps
###################################################


def append_avg_summary(dll_lines_dict):
    def average(lines_with_name, name):
        num = sum(map(lambda x: float(x["data"][name]), lines_with_name))
        denom = len(lines_with_name)
        avg = num/denom
        return avg
    return generate_summary_lines(dll_lines_dict, average)

def append_lastval_summary(dll_lines_dict):
    def last(lines_with_name, name):
        last_value = max(lines_with_name, key=lambda x: float(x["timestamp"]))["data"][name]
        return float(last_value)
    return generate_summary_lines(dll_lines_dict, last)

def generate_summary_lines(dll_lines_dict, get_value):
    #summary

    for name in dll_lines_dict:
        # skip metrics that already have a summary value
        skip_name = False
        for line in dll_lines_dict[name]:
            if line["step"] == -1:
                skip_name = True
                break
        if skip_name:
            continue

        value = get_value(dll_lines_dict[name], name)
        last_ts = max(map(lambda x: float(x["timestamp"]), dll_lines_dict[name]))
        dll_lines_dict[name].append(
            {
                "timestamp": str(last_ts),
                "data": {
                    name: value
                    },
                "step": -1
            }
        )
    lines = []
    for name in dll_lines_dict:
        lines += dll_lines_dict[name]
    return lines

def process_time(mllog, metrics = time_metrics):
    """
    Calculate statistics for the time between pairs of tags. If tag1 == tag2, then
    Calculate statistics for the time between multiple instances of the same tag.
    Ignore cases where the same tag is printed twice to the log with the exact same
    timestep as that is likely just logging error. Eventually we should fix our
    logging and remove this exclusion
        """
    time_lines_dict = {}
    saved_timestamp = defaultdict()
    for _, _, name in metrics:
        saved_timestamp[name] = None
    for line in mllog:
        new_timestamp = line["time_ms"]
        for tag1, tag2, name in metrics:
            if tag1 == line['key']:
                if tag1 == tag2 and saved_timestamp[name] and new_timestamp > saved_timestamp[name]:
                    entry = {
                            "timestamp": str(new_timestamp),
                            "data": {
                                name: (new_timestamp-saved_timestamp[name]) / (1000 * 60)
                                },
                            "step": line["metadata"]["epoch_num"] if "epoch_num" in line["metadata"] else -1
                            }
                    if name in time_lines_dict:
                        time_lines_dict[name].append(entry)
                    else:
                        time_lines_dict[name] = [entry]
                saved_timestamp[name] = new_timestamp
            elif saved_timestamp[name] and tag2 == line['key']:
                entry = {
                        "timestamp": str(new_timestamp),
                        "data": {
                            name: (new_timestamp-saved_timestamp[name]) / (1000 * 60)
                            },
                        "step": line["metadata"]["epoch_num"]  if "epoch_num" in line["metadata"] else -1
                        }
                saved_timestamp[name] = None
                if name in time_lines_dict:
                    time_lines_dict[name].append(entry)
                else:
                    time_lines_dict[name] = [entry]

    # averages appended to end and returned
    return append_avg_summary(time_lines_dict)

def get_total_eval(time_lines):
    d_total_eval = 0
    max_timestamp = 0
    for line in time_lines:
        if "d_eval" in line["data"]:
            d_total_eval += line["data"]["d_eval"]
            if float(line["timestamp"]) > max_timestamp:
                max_timestamp = float(line["timestamp"])
    if d_total_eval > 0:
        time_lines.append({"timestamp": str(max_timestamp),
                           "data": {"d_total_eval": d_total_eval},
                           "step": -1})
    return time_lines


def process_accuracy(name, mllog):
    accuracy_lines_dict = {}
    for json in mllog:
        if json["key"] == "eval_accuracy":
            if 'object_detection' in name:
                entry1 = {
                        "timestamp": str(json["time_ms"]),
                        "data": {
                            "d_accuracy": json["value"]["BBOX"]
                            },
                        "step": json["metadata"]["epoch_num"]
                        }
                if "d_accuracy" in accuracy_lines_dict:
                    accuracy_lines_dict["d_accuracy"].append(entry1)
                else:
                    accuracy_lines_dict["d_accuracy"] = [entry1]

                entry2 = {
                        "timestamp": str(json["time_ms"]),
                        "data": {
                            "d_accuracy2":json["value"]["SEGM"]
                            },
                        "step": json["metadata"]["epoch_num"]
                        }

                if "d_accuracy2" in accuracy_lines_dict:
                    accuracy_lines_dict["d_accuracy2"].append(entry2)
                else:
                    accuracy_lines_dict["d_accuracy2"] = [entry2]
            else:
                entry = {
                        "timestamp": str(json["time_ms"]),
                        "data": {
                            "d_accuracy": json["value"]
                            },
                        "step": json["metadata"]["epoch_num"]
                        }
                if "d_accuracy" in accuracy_lines_dict:
                    accuracy_lines_dict["d_accuracy"].append(entry)
                else:
                    accuracy_lines_dict["d_accuracy"] = [entry]

    # averages appended to end and returned
    return append_lastval_summary(accuracy_lines_dict)

def prune_log(stats_lines_dict, stats):
    def prune(log, stat, ratio):
        pruned_log = []
        pruned_log.append(log[0])
        tally = 0
        while (tally < len(log) - ratio):
            tally += ratio
            pruned_log.append(log[floor(tally)])
        return pruned_log

    if stats_lines_dict:
        num_lines = len(stats_lines_dict.keys()) * len(stats_lines_dict[stats[0]])
        ratio = num_lines/float(MAX_STATS_LINES)

        if ratio > 1:
            pruned_dict = {}
            for stat in stats:
                pruned_log = prune(stats_lines_dict[stat], stat, ratio)
                pruned_dict[stat] = pruned_log
            return pruned_dict

    return stats_lines_dict


def process_tracked_stats(mllog):
    stats_lines_dict = {}
    stats = None
    for json in mllog:
        if json["key"] == "tracked_stats":
            stats = [key for key in json["value"]]
            for stat in stats:
                entry = {
                        "timestamp": str(json["time_ms"]),
                        "data": {
                            stat : json["value"][stat]
                            },
                        "step": json["metadata"]["step"]
                        }
                if stat in stats_lines_dict:
                    stats_lines_dict[stat].append(entry)
                else:
                    stats_lines_dict[stat] = [entry]

    stats_lines_dict = prune_log(stats_lines_dict, stats)

    # averages appended to end and returned
    return append_avg_summary(stats_lines_dict)

###################################################
# ENV IMPORT
###################################################

def env():

    keys = [
        "BENCHMARK",
        "BENCHMARK_PATH",
        "BUILD",
        "BUILD_REPO",
        "BUILD_REPO_COMMIT",
        "BUILD_REPO_COMMIT_TS",
        "CI_COMMIT_REF_NAME",
        "CI_COMMIT_SHA",
        "CI_JOB_ID",
        "CI_JOB_NAME",
        "CI_PIPELINE_ID",
        "CLUSTER",
        "CUBLAS_CUSTOM_VERSION",
        "CUBLAS_VERSION",
        "CUDA_DRIVER_VERSION",
        "CUDA_VERSION",
        "CUDNN_CUSTOM_VERSION",
        "CUDNN_VERSION",
        "CUFFT_VERSION",
        "CURAND_VERSION",
        "CUSOLVER_VERSION",
        "CUSPARSE_VERSION",
        "DALI_BUILD",
        "DALI_VERSION",
        "DATESTAMP",
        "DEST_IMAGE",
        "DEST_IMAGE_VERSIONED",
        "DGXNGPU",
        "DGXNNODES",
        "FRAMEWORK",
        "FRAMEWORK_VARIANCE",
        "GPUS_PER_NODE",
        "NCCL_CUSTOM_VERSION",
        "NCCL_VERSION",
        "NIGHTLY_PIPE",
        "NVIDIA_BUILD_ID",
        "NVIDIA_PIPELINE_ID",
        "NVJPEG_VERSION",
        "PARENT_PIPELINE_ID"
        "PYTORCH_BUILD_VERSION",
        "PYTORCH_VERSION",
        "QA_IMAGE",
        "QA_IMAGE_VERSIONED",
        "SLURM_JOBID",
        "SLURM_NODELIST",
        "SLURM_WALLTIME",
        "TENSORFLOW_BUILD_VERSION",
        "TENSORFLOW_VERSION",
        "TEST_PASS",
        "TESTNAME",
        "UPDATE_LIBS",
        "UPDATE_REGRESSOR_LIBS",
    ]

    env = {k: os.environ.get(k, "") for k in keys}

    return env
