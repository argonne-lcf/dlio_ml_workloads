# The MIT License (MIT)
#
# Modifications Copyright (c) 2020-2022 NVIDIA CORPORATION. All rights reserved.
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
                                                                              

import os
import mmap
import io
import sys
import glob
import h5py as h5
import numpy as np
import concurrent.futures as cf
import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# we need cupy
import cupy as cp

class NumpyExternalGPUSource(object):

    def init_files(self):
        # determine shard sizes etc:
        # shard the bulk first
        num_files = len(self.data_files)
        num_files_per_shard = num_files // self.num_shards
        bulk_start = self.shard_id * num_files_per_shard
        bulk_end = bulk_start + num_files_per_shard
        
        # get the remainder now
        rem_start = self.num_shards * num_files_per_shard
        rem_end = num_files    

        # compute the chunked list
        self.data_files_chunks = []
        self.label_files_chunks = [] 
        for _ in range(self.oversampling_factor):
            
            # shuffle list
            perm = self.rng.permutation(range(num_files))
            data_files = np.array(self.data_files)[perm]
            label_files = np.array(self.label_files)[perm]

            # chunk: bulk
            data_files_chunk = data_files[bulk_start:bulk_end]
            label_files_chunk = label_files[bulk_start:bulk_end]

            # chunk: remainder
            data_rem = data_files[rem_start:rem_end]
            label_rem = label_files[rem_start:rem_end]
            if (self.shard_id < data_rem.shape[0]):
                np.append(data_files_chunk, data_rem[self.shard_id:self.shard_id+1], axis=0)
                np.append(label_files_chunk, label_rem[self.shard_id:self.shard_id+1], axis=0)

            self.data_files_chunks.append(data_files_chunk)
            self.label_files_chunks.append(label_files_chunk)

        return

            
    def start_prefetching(self):
        
        # we don't need to start prefetching again if it is already running
        # or if we do not cache data
        if self.prefetching_started or (not self.cache_data):
            return
        
        # flatten data lists: warning, unique sorts the data. We need to undo the sort
        # we need to make sure we return unique elements but maintain original ordering
        # data
        data_files = np.concatenate(self.data_files_chunks, axis = 0)
        data_files = data_files[np.sort(np.unique(data_files, return_index=True)[1])].tolist()
        # label
        label_files = np.concatenate(self.label_files_chunks, axis = 0)
        label_files = label_files[np.sort(np.unique(label_files, return_index=True)[1])].tolist()

        # allocate buffers
        self.data_buffers = [cp.empty(self.data_shape, dtype=self.data_dtype) for _ in range(len(data_files))]
        self.label_buffers = [cp.empty(self.label_shape, dtype=self.label_dtype) for _ in range(len(label_files))]
        
        # if zip does not work here, there is a bug
        for data_file, data_buff, label_file, label_buff in zip(data_files, self.data_buffers, label_files, self.label_buffers):
            # data and label are still unique, so just append to the queue
            self.prefetch_queue[data_file] = self.process_pool.submit(self._prefetch_sample, data_file, data_buff)
            self.prefetch_queue[label_file] = self.process_pool.submit(self._prefetch_sample, label_file, label_buff)
            
        # mark prefetching as started
        self.prefetching_started = True

        return


    def finalize_prefetching(self):
        if not self.prefetching_started or (not self.cache_data):
            return

        for task in cf.as_completed(self.prefetch_queue.values()):
            filename, data = task.result()
            self.file_cache[filename] = data
            
        return

    
    def _check_prefetching(self):
        # iterate over queue once:
        rmkeys = []
        for key in self.prefetch_queue.keys():
            task = self.prefetch_queue[key]
            if not task.done():
                continue
            
            filename, data = task.result()
            self.file_cache[filename] = data
            rmkeys.append(key)

        for key in rmkeys:
            self.prefetch_queue.pop(key)

        # check if queue is empty after this:
        if not self.prefetch_queue:
            self.cache_ready = True
        
        return
    
    
    def __init__(self, data_files, label_files, batch_size, device, last_batch_mode = "drop",
                 num_shards = 1, shard_id = 0, oversampling_factor = 1, shuffle = False,
                 cache_data = False, num_threads = 4, seed = 333):

        # important parameters
        self.data_files = data_files
        self.label_files = label_files
        self.batch_size = batch_size
        self.last_batch_mode = last_batch_mode

        # sharding info
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.oversampling_factor = oversampling_factor
        self.shuffle = shuffle
        self.device = cp.cuda.Device(device.index)
        self.cache_data = cache_data
        self.seed = seed
        self.rng = np.random.default_rng(seed = self.seed)

        # file cache relevant stuff
        self.prefetch_queue = {}
        self.file_cache = {}
        self.process_pool = cf.ThreadPoolExecutor(max_workers = num_threads)
        self.prefetching_started = False
        self.cache_ready = False
        self.use_direct_io = True

        # running parameters
        self.chunk_idx = 0
        self.file_idx = 0

        # make sure device is used
        self.device.use()

        # init file lists
        self.init_files()

        # some buffer for double buffering
        # determine shapes first, then preallocate
        data = np.load(self.data_files_chunks[0][0])
        self.data_shape, self.data_dtype = data.shape, data.dtype
        label = np.load(self.label_files_chunks[0][0])
        self.label_shape, self.label_dtype = label.shape, label.dtype
        # allocate buffers
        #self.data_batch = [ np.zeros(self.data_shape, dtype=self.data_dtype) for _ in range(self.batch_size) ]
        #self.label_batch = [ np.zeros(self.label_shape, dtype=self.label_dtype) for _ in range(self.batch_size) ] 

        
    def __iter__(self):
        self.chunk_idx = (self.chunk_idx + 1) % self.oversampling_factor
        self.file_idx = 0
        self.length = self.data_files_chunks[self.chunk_idx].shape[0]

        # set new lists
        self.data_files_current = self.data_files_chunks[self.chunk_idx]
        self.label_files_current = self.label_files_chunks[self.chunk_idx]
        
        # shuffle chunk ONLY if prefetching is done
        if self.shuffle and self.cache_ready:
            perm = self.rng.permutation(range(self.length))
            self.data_files_current = self.data_files_current[perm]
            self.label_files_current = self.label_files_current[perm]

        if self.prefetching_started and not self.cache_ready:
            self._check_prefetching()
            
        self.get_sample_handle = self._get_sample_cache if self.cache_ready else self._get_sample_test
            
        return self

    
    def _get_sample_cache(self, data_filename, label_filename, batch_id=0):
        return self.file_cache[data_filename], self.file_cache[label_filename]


    def _load_sample(self, filename, buff):
        
        torch.cuda.nvtx.range_push("NumpyExternalGPUSource::_load_sample")
        
        token = cp.load(filename)
        buff[...] = token[...]
            
        torch.cuda.nvtx.range_pop()
            
        return buff
    

    def _get_sample_test(self, data_filename, label_filename, batch_id=0):

        torch.cuda.nvtx.range_push("NumpyExternalGPUSource::_get_sample_test")
        
        #print(f"_get_sample_test for {data_filename}, {label_filename}")
        # fetch data
        if data_filename not in self.file_cache:
            #print(f"wait for {data_filename}")
            _, token = self.prefetch_queue[data_filename].result()
            self.file_cache[data_filename] = token
            self.prefetch_queue.pop(data_filename)
            
        data = self.file_cache[data_filename]

        # fetch label
        if label_filename not in self.file_cache:
            #print(f"wait for {label_filename}")
            _, token = self.prefetch_queue[label_filename].result()
            self.file_cache[label_filename] = token
            self.prefetch_queue.pop(label_filename)
                
        label = self.file_cache[label_filename]

        torch.cuda.nvtx.range_pop()
            
        return data, label
    
    
    def _prefetch_sample(self, filename, buff):
        data = self._load_sample(filename, buff)
        return filename, data
    
    
    def __next__(self):
        torch.cuda.nvtx.range_push("NumpyExternalGPUSource::__next__")
        # prepare empty batch
        data = []
        label = []
        
        # check if epoch ends here
        if self.file_idx >= self.length:
            raise StopIteration

        if ((self.file_idx + self.batch_size) >= self.length) and (self.last_batch_mode == "drop"):
            raise StopIteration
        elif (self.last_batch_mode == "partial"):
            batch_size_eff = min([self.length - self.file_idx, self.batch_size])
        else:
            batch_size_eff = self.batch_size

        #print("Get Batch")
            
        # fill batch
        for idb in range(batch_size_eff):

            data_filename = self.data_files_current[self.file_idx]
            label_filename = self.label_files_current[self.file_idx]
            data_token, label_token = self.get_sample_handle(data_filename, label_filename, idb)
            data.append(data_token)
            label.append(label_token)
            self.file_idx = self.file_idx + 1
            
        torch.cuda.nvtx.range_pop()

        #print("Done getting batch")
        
        return (data, label)


class CamDaliESGPUDataloader(object):

    def get_pipeline(self):
        self.data_size = 1152*768*16*4
        self.label_size = 1152*768*4
        
        pipeline = Pipeline(batch_size = self.batchsize, 
                            num_threads = self.num_threads, 
                            device_id = self.device.index,
                            seed = self.seed)
                                 
        with pipeline:
            # no_copy = True is only safe to use if data cache is enabled
            data, label = fn.external_source(source = self.extsource,
                                             device = "gpu",
                                             num_outputs = 2,
                                             cycle = "raise",
                                             no_copy = self.cache_data,
                                             parallel = False)
            
            # normalize data
            data = fn.normalize(data, 
                                device = "gpu",
                                mean = self.data_mean,
                                stddev = self.data_stddev,
                                scale = 1.,
                                bytes_per_sample_hint = self.data_size)
            
            # cast label to long
            label = fn.cast(label,
                            device = "gpu",
                            dtype = types.DALIDataType.INT64,
                            bytes_per_sample_hint = self.label_size)
                            
            if self.transpose:
                data = fn.transpose(data,
                                    device = "gpu",
                                    perm = [2, 0, 1],
                                    bytes_per_sample_hint = self.data_size)

            pipeline.set_outputs(data, label)
            
        return pipeline
            


    def init_files(self, root_dir, prefix_data, prefix_label, statsfile, file_list_data = None, file_list_label = None):
        self.root_dir = root_dir
        self.prefix_data = prefix_data
        self.prefix_label = prefix_label

        # get files
        # data
        if file_list_data is not None and os.path.isfile(os.path.join(root_dir, file_list_data)):
            with open(os.path.join(root_dir, file_list_data), "r") as f:
                token = f.readlines()
            self.data_files = sorted([os.path.join(root_dir, x.strip()) for x in token])
        else:
            self.data_files = sorted(glob.glob(os.path.join(self.root_dir, self.prefix_data)))
        # label
        if file_list_label is not None and os.path.isfile(os.path.join(root_dir, file_list_label)):
            with open(os.path.join(root_dir, file_list_label), "r") as f:
                token = f.readlines()
            self.label_files = sorted([os.path.join(root_dir, x.strip()) for x in token])
        else:
            self.label_files = sorted(glob.glob(os.path.join(self.root_dir, self.prefix_label)))

        # get shapes
        self.data_shape = np.load(self.data_files[0]).shape
        self.label_shape = np.load(self.label_files[0]).shape

        # open statsfile
        with h5.File(statsfile, "r") as f:
            data_mean = f["climate"]["minval"][...]
            data_stddev = (f["climate"]["maxval"][...] - data_mean)
            
        #reshape into broadcastable shape: channels first
        self.data_mean = np.reshape( data_mean, (1, 1, data_mean.shape[0]) ).astype(np.float32)
        self.data_stddev = np.reshape( data_stddev, (1, 1, data_stddev.shape[0]) ).astype(np.float32)

        # clean up old iterator
        if self.iterator is not None:
            del(self.iterator)
            self.iterator = None
        
        # clean up old pipeline
        if self.pipeline is not None:
            del(self.pipeline)
            self.pipeline = None

        # io devices
        self.io_device = "gpu" if self.read_gpu else "cpu"

        # create ES
        self.extsource = NumpyExternalGPUSource(self.data_files, self.label_files, self.batchsize,
                                                device = self.device,
                                                last_batch_mode = "partial" if self.is_validation else "drop",
                                                num_shards = self.num_shards, shard_id = self.shard_id,
                                                oversampling_factor = self.oversampling_factor, shuffle = self.shuffle,
                                                cache_data = self.cache_data, num_threads = self.num_es_threads, seed = self.seed)
        
        # set up pipeline
        self.pipeline = self.get_pipeline()
       
        # build pipes
        self.global_size = len(self.data_files)
        self.pipeline.build()


    def start_prefetching(self):
        self.extsource.start_prefetching()

        
    def finalize_prefetching(self):
        self.extsource.finalize_prefetching()

        
    def __init__(self, root_dir, prefix_data, prefix_label, statsfile,
                 batchsize, file_list_data = None, file_list_label = None,
                 num_threads = 1, device = torch.device("cpu"),
                 num_shards = 1, shard_id = 0,
                 shuffle_mode = None, oversampling_factor = 1,
                 is_validation = False,
                 lazy_init = False, transpose = True, augmentations = [],
                 use_mmap = True, read_gpu = False, seed = 333):
    
        # read filenames first
        self.batchsize = batchsize
        self.num_threads = num_threads
        self.num_es_threads = 2
        self.device = device
        self.io_device = "gpu" if read_gpu else "cpu"
        self.use_mmap = use_mmap
        self.shuffle_mode = shuffle_mode
        self.read_gpu = read_gpu
        self.pipeline = None
        self.iterator = None
        self.lazy_init = lazy_init
        self.transpose = transpose
        self.augmentations = augmentations
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.is_validation = is_validation
        self.seed = seed
        self.epoch_size = 0
        
        # shuffle mode:
        if self.shuffle_mode is not None:
            self.shuffle = True
            self.stick_to_shard = False
        else:
            self.shuffle = False
            self.stick_to_shard = True

        # ES perf opts
        self.cache_data = True
        self.oversampling_factor = oversampling_factor
            
        # get rng
        self.rng = np.random.default_rng(self.seed)
            
        # init files
        self.init_files(root_dir, prefix_data, prefix_label,
                        statsfile, file_list_data, file_list_label)
        
        self.iterator = DALIGenericIterator([self.pipeline], ['data', 'label'], auto_reset = True,
                                            size = -1,
                                            last_batch_policy = LastBatchPolicy.PARTIAL if self.is_validation else LastBatchPolicy.DROP,
                                            prepare_first_batch = False)

        self.epoch_size = self.pipeline.epoch_size()
        

    @property
    def shapes(self):
        return self.data_shape, self.label_shape

    
    def __iter__(self):
        #self.iterator.reset()
        for token in self.iterator:
            data = token[0]['data']
            label = token[0]['label']
            
            yield data, label, ""
