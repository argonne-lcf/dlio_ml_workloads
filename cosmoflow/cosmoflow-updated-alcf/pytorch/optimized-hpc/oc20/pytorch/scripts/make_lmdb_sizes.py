# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

"""
This script provides the functionality to generate metadata.npz files necessary
for load_balancing the DataLoader.

"""


import argparse
import multiprocessing as mp
import os
import warnings

import numpy as np
from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset
from tqdm import tqdm


def get_data(index):
    data = dataset[index]
    natoms = data.natoms
    neighbors = None
    if hasattr(data, "edge_index"):
        neighbors = data.edge_index.shape[1]

    return index, natoms, neighbors


def main(args):
    path = args.data_path
    global dataset
    if os.path.isdir(path):
        dataset = TrajectoryLmdbDataset({"src": path})
        outpath = os.path.join(path, "metadata.npz")
    elif os.path.isfile(path):
        dataset = SinglePointLmdbDataset({"src": path})
        outpath = os.path.join(os.path.dirname(path), "metadata.npz")

    indices = range(len(dataset))

    pool = mp.Pool(args.num_workers)
    outputs = list(tqdm(pool.imap(get_data, indices), total=len(indices)))

    indices = []
    natoms = []
    neighbors = []
    for i in outputs:
        indices.append(i[0])
        natoms.append(i[1])
        neighbors.append(i[2])

    _sort = np.argsort(indices)
    sorted_natoms = np.array(natoms, dtype=np.int32)[_sort]
    if None in neighbors:
        warnings.warn(f"edge_index information not found, {outpath} only supports atom-wise load balancing.")
        np.savez(outpath, natoms=sorted_natoms)
    else:
        sorted_neighbors = np.array(neighbors, dtype=np.int32)[_sort]
        np.savez(outpath, natoms=sorted_natoms, neighbors=sorted_neighbors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Path to S2EF directory or IS2R* .lmdb file",
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Num of workers to parallelize across",
    )
    args = parser.parse_args()
    main(args)
