#!/usr/bin/python3
import argparse
import json
import glob
import sys
import os
import re
import statistics
import mlperf_post_process_lib as lib
import datetime
import sys
from statistics import mean
from collections import defaultdict
from dllogger.dllogger import JSONStreamBackend, LogLevel

def process_arguments():
    parser = argparse.ArgumentParser(description="Process MLPerf Benchmark results from a log file. Script will look for a"\
        "entry.json file one directory above the result logfiles, and if found, will extract system info from it.")
    parser.add_argument('--file', type=lib.file_path, \
        help="Input file to process with MLPerf compliance package logging info. Assumes that the path to the file is of " +\
        "the format .../benchmark_name/resultfile")
    parser.add_argument('--logdir', help="log directory so update_libs can be found")
    parser.add_argument('--env_json_path', type=argparse.FileType('w'), help="path of env json that is output")
    parser.add_argument('--dll_path', help="path of dllog object that is output")
    parser.add_argument('--benchmark', help="INPUT $BENCHMARK eg. image_classification")
    parser.add_argument('--system_config', help="Input $DGXSYSTEM eg. DGXA100_multi_2x8x204")
    args = parser.parse_args()
    return args

def parse_log(args):

    if not args.file:
        return

    times = {}
    settings = {}
    throughput = {}
    hyperparam_metrics = {}

    benchmark = args.benchmark
    config = args.system_config
    hyperparam_metrics = lib.get_hyperparam_metrics(benchmark)

    # --------------------------------------------------------------------------------
    # Get MLLog
    fh = open (args.file, 'r', encoding='ISO-8859-1')
    mllog = lib.extract_mllog(fh)
    if len(mllog) == 0:
        return None
    earliest_ts = min(map(lambda x: float(x["time_ms"]), mllog))

    # --------------------------------------------------------------------------------
    # DLLog Parameters
    parameters_json = {}

    settings = lib.process_params(lib.param_metrics, mllog)
    #if LR is a string of list, convert to a real list
    try:
        if "d_LR" in settings:
            settings['d_LR'] = json.loads(settings['d_LR'])
    except:
        pass
    hyperparam_settings = lib.process_params(hyperparam_metrics, mllog)
    submission = lib.process_submission(mllog, benchmark, config)
    # collect process type(horovod/device) details for image_classification benchmark
    multiprocess_dict = lib.multiprocess(args.file, mllog)

    parameters_json.update(settings)
    parameters_json.update(hyperparam_settings)
    parameters_json.update(multiprocess_dict)
    parameters_json.update(submission)

    # --------------------------------------------------------------------------------
    # DLLog Log
    metrics_lines = []

    time_lines = lib.process_time(mllog)
    time_lines = lib.get_total_eval(time_lines)
    accuracy_lines = lib.process_accuracy(args.benchmark, mllog)
    stats_lines = lib.process_tracked_stats(mllog)

    metrics_lines += time_lines
    metrics_lines += accuracy_lines
    metrics_lines += stats_lines

    # --------------------------------------------------------------------------------
    # Env Import for later
    env_json = {}

    env = lib.env()
    slurm = lib.process_job_data(benchmark, fh)

    # Gettting the data from update library stage in gitlab
    json_files_for_update_lib = glob.glob(args.logdir + "/*update*json")
    update_lib_data = {}
    for file in json_files_for_update_lib:
        if benchmark in file:
            if 'ssd_mxnet' in args.file and 'ssd_mxnet' not in file:
                continue
            elif 'ssd_mxnet' not in args.file and benchmark not in file:
                continue
            print("using file for lib versions:", file)
            with open(file) as f:
                update_lib_data = json.load(f)

    env_json.update(env)
    env_json.update(slurm)
    env_json.update(update_lib_data)

    return (earliest_ts, parameters_json, metrics_lines, env_json)


################################### main program starts here ####################

if __name__=='__main__':

    args = process_arguments()
    ret = parse_log(args)
    if not ret is None:
        earliest_ts, parameters_json, metrics_lines, env_json = ret
        logger = JSONStreamBackend(LogLevel.INFO, args.dll_path)

        # generate param line
        # TS is in MS
        start_timestamp = datetime.datetime.fromtimestamp(earliest_ts/1000)
        logger.log(timestamp=start_timestamp, elapsedtime=0, step="PARAMETER", data=parameters_json)

        # generate log lines
        for line in metrics_lines:
            elapsedtime = datetime.datetime.fromtimestamp(float(line["timestamp"])/1000).timestamp()-start_timestamp.timestamp()
            logger.log(timestamp=datetime.datetime.fromtimestamp(float(line["timestamp"])/1000), elapsedtime=elapsedtime, step=() if line["step"] == -1 else line["step"], data=line["data"])

        # Add slurm walltime
        logger.log(timestamp=start_timestamp, elapsedtime=0, step=(), data= {"SLURM_WALLTIME": float(env_json["SLURM_WALLTIME"]) / 60})

        # generate env json
        json.dump(env_json, args.env_json_path)
        # print(json.dumps(env_json, indent=4, sort_keys=True))
    else:
        exit(1)
