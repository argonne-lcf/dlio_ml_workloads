"""CosmoFlow dataset specification"""

# System imports
import os
import logging
import glob
from functools import partial

# External imports
import numpy as np
from numpy.lib import stride_tricks
import tensorflow as tf
import horovod.tensorflow.keras as hvd
try:
    from mlperf_logging import mllog
    have_mlperf_logging = True
except ImportError:
    have_mlperf_logging = False

# Local imports
import utils.distributed
from utils.staging import stage_files

import pathlib
from typing import Union, List, Tuple

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as dali_fn
import nvidia.dali.math as dali_math
import nvidia.dali.types as dali_types

import nvidia.dali.plugin.tf

FileListType = Tuple[Union[pathlib.Path, List[str]], 
                     Union[pathlib.Path, List[str]]]

def get_dali_pipeline(file_root: pathlib.Path, file_list: FileListType, *, 
                      dont_use_mmap: bool = True, num_shards: int = 1, 
                      shard_id: int = 0, apply_log: bool = False, batch_size: int = 1,
                      dali_threads: int = 1, device_id: int = 0) -> Pipeline:
    if isinstance(file_list[0], pathlib.Path):
        with open(file_list[0], "r") as files_data:
            file_list_data = files_data.readlines()
    else:
        file_list_data = file_list[0]

    if isinstance(file_list[1], pathlib.Path):
        with open(file_list[1], "r") as files_label:
            file_list_label = files_label.readlines()
    else:
        file_list_label = file_list[1]

    SAMPLE_SIZE_DATA = 4*128*128*128*4
    SAMPLE_SIZE_LABEL = 4*4

    pipeline = Pipeline(batch_size=batch_size,
                        num_threads=dali_threads, 
                        device_id=device_id)
    with pipeline:
        numpy_reader = dali_fn.readers.numpy(bytes_per_sample_hint=SAMPLE_SIZE_DATA / 2,
                                            dont_use_mmap=dont_use_mmap,
                                            file_root=str(file_root),
                                            files=file_list_data,
                                            num_shards=num_shards,
                                            shard_id=shard_id,
                                            stick_to_shard=True)
        label_reader = dali_fn.readers.numpy(bytes_per_sample_hint=SAMPLE_SIZE_LABEL,
                                            dont_use_mmap=dont_use_mmap,
                                            file_root=str(file_root),
                                            files=file_list_label,
                                            num_shards=num_shards,
                                            shard_id=shard_id,
                                            stick_to_shard=True)

        feature_map = dali_fn.cast(numpy_reader.gpu(), dtype=dali_types.FLOAT,
                                   bytes_per_sample_hint=SAMPLE_SIZE_DATA)
        if apply_log:
            feature_map = dali_math.log(feature_map + 1.0)
        else:
            feature_map = feature_map / dali_fn.reductions.mean(feature_map)
        pipeline.set_outputs(feature_map, label_reader.gpu())
    pipeline.build()
    return pipeline


def construct_dataset(file_dir, n_samples, batch_size, n_epochs,
                      sample_shape, samples_per_file=1, n_file_sets=1,
                      shard=0, n_shards=1, apply_log=False,
                      randomize_files=False, shuffle=False,
                      shuffle_buffer_size=0, n_parallel_reads=4, prefetch=4,
                      compression=None, device_id=0):
    """This function takes a folder with files and builds the TF dataset.

    It ensures that the requested sample counts are divisible by files,
    local-disks, worker shards, and mini-batches.
    """

    if n_samples == 0:
        return None, 0

    # Ensure samples divide evenly into files * local-disks * worker-shards * batches
    n_divs = samples_per_file * n_file_sets * n_shards * batch_size
    if (n_samples % n_divs) != 0:
        logging.error('Number of samples (%i) not divisible by %i '
                      'samples_per_file * n_file_sets * n_shards * batch_size',
                      n_samples, n_divs)
        raise Exception('Invalid sample counts')

    # Number of files and steps
    n_files = n_samples // (samples_per_file * n_file_sets)
    n_steps = n_samples // (n_file_sets * n_shards * batch_size)

    # Find the files
    with open(os.path.join(file_dir, "files_data.lst"), "r") as f:
        token = f.readlines()
    data_filenames = [x.strip() for x in sorted(token)]
    with open(os.path.join(file_dir, "files_label.lst"), "r") as f:
        token = f.readlines()
    label_filenames = [x.strip() for x in sorted(token)]
    
    assert (0 <= n_files) and (n_files <= len(data_filenames)), (
        'Requested %i files, %i available' % (n_files, len(data_filenames)))
    if randomize_files:
        permutation = np.random.permute(range(len(data_filenames)))
        data_filenames = data_filenames[permutation]
        label_filenames = label_filenames[permutation]

    # reduce
    data_filenames = data_filenames[:n_files]
    label_filenames = label_filenames[:n_files]

    # Define the dataset from the list of sharded, shuffled files
    target_shape = [4]

    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.autotune = False

    with tf.device("/gpu:0"):
        dataset = nvidia.dali.plugin.tf.DALIDataset(get_dali_pipeline(pathlib.Path(file_dir), 
                                                                      (data_filenames, label_filenames),
                                                                      dont_use_mmap=True, num_shards=n_shards,
                                                                      shard_id=shard, apply_log=apply_log, batch_size=batch_size,
                                                                      dali_threads=n_parallel_reads, device_id=device_id),
                                                    device_id=device_id,
                                                    batch_size=batch_size,
                                                    output_shapes=(tuple([batch_size] + sample_shape), 
                                                                tuple([batch_size] + target_shape)),
                                                    output_dtypes=(tf.float32, tf.float32),
                                                    fail_on_device_mismatch=True)
        dataset = dataset.with_options(options)
    return dataset, n_steps

def get_datasets(data_dir, sample_shape, n_train, n_valid,
                 batch_size, n_epochs, dist, samples_per_file=1,
                 shuffle_train=True, shuffle_valid=False,
                 shard=True, stage_dir=None, apply_log=False,
                 **kwargs):
    """Prepare TF datasets for training and validation.

    This function will perform optional staging of data chunks to local
    filesystems. It also figures out how to split files according to local
    filesystems (if pre-staging) and worker shards (if sharding).

    Returns: A dict of the two datasets and step counts per epoch.
    """

    # MLPerf logging
    if dist.rank == 0 and have_mlperf_logging:
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=batch_size*dist.size)
        mllogger.event(key=mllog.constants.TRAIN_SAMPLES, value=n_train)
        mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=n_valid)
    data_dir = os.path.expandvars(data_dir)

    # Synchronize before local data staging
    utils.distributed.barrier()

    # Local data staging
    if dist.rank == 0 and have_mlperf_logging:
        mllogger.start(key='staging_start')

    if stage_dir is not None:
        staged_files = True
        # Stage training data
        stage_files(os.path.join(data_dir, 'train'),
                    os.path.join(stage_dir, 'train'),
                    n_files=n_train, rank=dist.rank, size=dist.size)
        # Stage validation data
        stage_files(os.path.join(data_dir, 'validation'),
                    os.path.join(stage_dir, 'validation'),
                    n_files=n_valid, rank=dist.rank, size=dist.size)
        data_dir = stage_dir
    else:
        staged_files = False

    # Barrier for workers to be done transferring
    utils.distributed.barrier()
    if dist.rank == 0 and have_mlperf_logging:
        mllogger.end(key='staging_stop')

    # Determine number of staged file sets and worker shards
    n_file_sets = (dist.size // dist.local_size) if staged_files else 1
    if shard and staged_files:
        shard, n_shards = dist.local_rank, dist.local_size
    elif shard and not staged_files:
        shard, n_shards = dist.rank, dist.size
    else:
        shard, n_shards = 0, 1

    device_id = dist.local_rank
    # Construct the training and validation datasets
    dataset_args = dict(batch_size=batch_size, n_epochs=n_epochs,
                        sample_shape=sample_shape, samples_per_file=samples_per_file,
                        n_file_sets=n_file_sets, shard=shard, n_shards=n_shards,
                        apply_log=apply_log, device_id=device_id, **kwargs)
    train_dataset, n_train_steps = construct_dataset(
        file_dir=os.path.join(data_dir, 'train'),
        n_samples=n_train, shuffle=shuffle_train, **dataset_args)
    valid_dataset, n_valid_steps = construct_dataset(
        file_dir=os.path.join(data_dir, 'validation'),
        n_samples=n_valid, shuffle=shuffle_valid, **dataset_args)

    if shard == 0:
        if staged_files:
            logging.info('Using %i locally-staged file sets', n_file_sets)
        logging.info('Splitting data into %i worker shards', n_shards)
        n_train_worker = n_train // (samples_per_file * n_file_sets * n_shards)
        n_valid_worker = n_valid // (samples_per_file * n_file_sets * n_shards)
        logging.info('Each worker reading %i training samples and %i validation samples',
                     n_train_worker, n_valid_worker)

    if dist.rank == 0:
        logging.info('Data setting n_train: %i', n_train)
        logging.info('Data setting n_valid: %i', n_valid)
        logging.info('Data setting batch_size: %i', batch_size)
        for k, v in kwargs.items():
            logging.info('Data setting %s: %s', k, v)

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train_steps=n_train_steps, n_valid_steps=n_valid_steps)
