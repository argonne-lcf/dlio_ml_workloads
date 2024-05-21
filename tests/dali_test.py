from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import os
import time
from pfw_utils.utility import Profile, PerfTrace

# To run with different data, see documentation of nvidia.dali.fn.readers.file
# points to https://github.com/NVIDIA/DALI_extra
#data_root_dir = os.environ['DALI_EXTRA_PATH']
images_dir = "/eagle/datascience/ImageNet/ILSVRC/Data/CLS-LOC/"
PerfTrace.initialize_log("./results/trace.pfw", images_dir, process_id=0)

def loss_func(pred, y):
    pass


def model(x):
    pass


def backward(loss, model):
    pass


@pipeline_def(num_threads=16, device_id=0)
def get_dali_pipeline():
    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=True, name="Reader", dont_use_mmap=True)
    # decode data on the GPU
    images = fn.decoders.image_random_crop(
        images, device="mixed", output_type=types.RGB)
    # the rest of processing happens on the GPU as well
    images = fn.resize(images, resize_x=256, resize_y=256)
    images = fn.crop_mirror_normalize(
        images,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip())
    return images, labels


train_data = DALIGenericIterator(
    [get_dali_pipeline(batch_size=4)],
    ['data', 'label'],
    reader_name='Reader'
)
from tqdm import tqdm 

dlp=Profile(cat="train", name="IO")
for i, data in tqdm(enumerate(train_data)):
    x, y = data[0]['data'], data[0]['label']
    with Profile(cat="train", name="compute"):
        time.sleep(0.001)
        pred = model(x)
        loss = loss_func(pred, y)
        backward(loss, model)
