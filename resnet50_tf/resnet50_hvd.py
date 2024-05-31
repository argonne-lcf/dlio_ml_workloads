# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from mpi4py import MPI
import argparse
import os
import numpy as np
import time
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.keras import applications
hvd.init()
from pfw_utils.utility import Profile, PerfTrace, Metric
import logging, sys
log = logging.getLogger('ResNet50')
log.setLevel(logging.DEBUG)
# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=400,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--steps', type=int, default=100,
                    help='number of batches per benchmark iteration')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--num_computation_threads', type=int, default=4,
                    help='number of computation threads')                    
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--data_folder', type=str, default="/eagle/datasets/ImageNet/tfrecords")                    
parser.add_argument("--output_folder", default='outputs', type=str)
parser.add_argument("--transfer_size", default=262144, type=int)
parser.add_argument("--datagen", default='synthetic')
args = parser.parse_args()
args.cuda = not args.no_cuda
hvd.init()
pfwlogger = PerfTrace.initialize_log(f"{args.output_folder}/trace-{hvd.rank()}-of-{hvd.size()}.pfw", os.path.abspath(args.data_folder), process_id=hvd.rank())    
dlp = Profile("RESNET50")


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(f"{args.output_folder}/resnet50_tf.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.ERROR)
ch.setFormatter(formatter)
log.addHandler(ch)
# Horovod: initialize Horovod.


# Horovod: pin GPU to be used to process local rank (one GPU per process)
if args.cuda:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set up standard model.
model = getattr(applications, args.model)(weights=None)
opt = tf.optimizers.SGD(0.01)

data = tf.random.uniform([args.batch_size, 224, 224, 3])
target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)

def get_datagen():
    MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    def normalize_image(image):
        image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
        image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
        return image
    if args.datagen == 'image':
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.125),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
            tf.keras.layers.experimental.preprocessing.RandomTranslation(0.3, 0.3),
        ])
        flist = glob.glob("%s/*/*"%args.data_folder)
        dirs = glob.glob("%s/*"%args.data_folder)
        classes = tf.constant([d.split("/")[-1] for d in dirs])
        ds = tf.data.Dataset.from_tensor_slices(flist)
        ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
        @tf.function
        def read_file(file_path):
            label = tf.strings.split(file_path, os.sep)[-2]
            image = tf.io.read_file(file_path)
            label = tf.where(tf.cast(tf.equal(classes, label), tf.uint8))
            return image, label
        @tf.function
        def preprocess(image):
            image = tf.io.decode_jpeg(image, channels=3)
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [224, 224])
            image = tf.image.random_flip_left_right(image)
            return normalize_image(image)            
        ds = ds.map(read_file, num_parallel_calls=args.num_workers)
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=args.num_computation_threads)
        ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    elif args.datagen=="tfrecord":
        @dlp.log
        def pass_fun(x):
            features = {
                'image/class/label': tf.io.FixedLenFeature([], tf.int64),
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/class/text':tf.io.FixedLenFeature([], tf.string)
            }
            parsed = tf.io.parse_single_example(x, features)
            image = tf.io.decode_jpeg(parsed["image/encoded"], channels=3)
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [224, 224])
            image = tf.image.random_flip_left_right(image)
            image = normalize_image(image)
            name = parsed['image/class/text']
            lab = parsed["image/class/label"]
            return image, lab
        file_names = glob.glob(f"{args.data_folder}/*")        
        ds = tf.data.TFRecordDataset(filenames=file_names, buffer_size=args.transfer_size, num_parallel_reads=args.num_workers)
        ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
        ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        ds = ds.map(pass_fun, num_parallel_calls = args.num_computation_threads)        
    else:
        x = np.random.random((4000, 224, 224, 3))
        y = np.random.random((4000, 1))
        X = tf.data.Dataset.from_tensor_slices(x)
        Y = tf.data.Dataset.from_tensor_slices(y)
        ds = tf.data.Dataset.zip((X, Y)).repeat().batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
@dlp.log
def benchmark_step(a, b, first_batch):
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    # Horovod: use DistributedGradientTape
    with tf.GradientTape() as tape:
        probs = model(a, training=True)
        loss = tf.losses.sparse_categorical_crossentropy(b, probs)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape, compression=compression)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

device = 'GPU' if args.cuda else 'CPU'
if hvd.rank()==0:
    log.info('Model: %s' % args.model)
    log.info('Batch size: %d' % args.batch_size)
    log.info('Number of %ss: %d' % (device, hvd.size()))
import glob


metric = Metric(args.batch_size, log_dir=args.output_folder)

#ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=True)
#ds = ds.apply(
#        tf.data.experimental.parallel_interleave(
#            tf.data.TFRecordDataset,
#            cycle_length=args.num_workers,
#            prefetch_input_elements=args.batch_size))
ds = get_datagen()

with tf.device(device):
    # Warm-up
    if hvd.rank() == 0:
        log.info('Running warmup...')
    for a, b in ds.take(1):
        #benchmark_step(a, b, first_batch=True)
        benchmark_step(a, b, first_batch=False)
    # Benchmark
    if hvd.rank()==0:
        log.info('Running benchmark...')
    img_secs = []
    for e in range(args.epochs):
        t = time.time()
        metric.start_epoch(e)
        metric.start_loading(0)
        step = 0
        for a, b in ds.take(args.steps):
            metric.end_loading(step)        
            metric.start_compute(step)
            with Profile(name="compute", cat='train'):
                benchmark_step(a, b, first_batch=False)
            metric.end_compute(step)
            step += 1
            metric.start_loading(step)            
        metric.end_epoch(e)
        t = time.time() -t
        img_sec = args.batch_size * args.steps / t
        if hvd.rank()==0:
            log.info('Iter #%d: %.1f img/sec per %s' % (e, img_sec, device))
            img_secs.append(img_sec)
    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    if hvd.rank()==0:
        log.info('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
        log.info('Total img/sec on %d %s(s): %.1f +-%.1f' %
            (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
pfwlogger.finalize()
