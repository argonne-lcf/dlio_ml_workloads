from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import numpy as np
import glob

def DaliDataLoader(data_dir, idx_dir, num_shards=1, shard_idx=0, batch_size=1, num_threads=4, device_id=0):
    data = sorted(glob.glob(f"{data_dir}/*"))
    idx = sorted(glob.glob(f"{idx_dir}/*"))
    data = data[shard_idx::num_shards]
    idx = data[shard_idx::num_shards]
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)

    with pipe:
        inputs = fn.readers.tfrecord(
            path=data,
            index_path=idx,
            features = {
                'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                'image/encoded': tfrec.FixedLenFeature([], tfrec.string, ""),
                'image/class/text':tfrec.FixedLenFeature([], tfrec.string, "")
            }
        )
        jpegs = inputs["image/encoded"]
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        resized = fn.resize(images, device="gpu", resize_shorter=256.0)
        output = fn.crop_mirror_normalize(
            resized,
            dtype=types.FLOAT,
            crop=(224, 224),
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        )
        pipe.set_outputs(output, inputs["image/class/text"])
    pipe.build()
    pipe_out = pipe.run()
    return pip_out

if __name__ == "__main__":
    data_dir = "/eagle/DLIO/datasets/resnet50/tfrecords/data/"
    idx_dir = "/eagle/DLIO/datasets/resnet50/tfrecords/idx/"
    loader = DaliDataLoader(data_dir, idx_dir, device_id=None)
    for d in loader:
        print(d)
