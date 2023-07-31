#!/bin/bash                                                                                                                                         
#PBS -S /bin/bash                                                                                                                                   
#PBS -l nodes=8:ppn=32
#PBS -l walltime=1:00:00                                                                                                                            
#PBS -M huihuo.zheng@anl.gov                                                                                                                        
#PBS -A datascience
#PBS -q debug-scaling
cd $PBS_O_WORKDIR
export DATA_SRC_DIR=/home/hzheng/datascience_grand/mlperf_hpc/hpc-nvidia/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_v2/
export DATA_DST_DIR=/home/hzheng/datascience_grand/mlperf_hpc/hpc-nvidia/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_npy_v2/
module load conda/2022-07-19; conda activate

aprun -n 256 -N 32 python tools/convert_tfrecord_to_numpy.py -i ${DATA_SRC_DIR}/train -o ${DATA_DST_DIR}/train -c gzip
aprun -n 256 -N 32 python tools/convert_tfrecord_to_numpy.py -i ${DATA_SRC_DIR}/validation -o ${DATA_DST_DIR}/validation -c gzip

#ls -1 ${DATA_DST_DIR}/train | grep "_data.npy" | sort > ${DATA_DST_DIR}/train/files_data.lst
#ls -1 ${DATA_DST_DIR}/validation | grep "_data.npy" | sort > ${DATA_DST_DIR}/validation/files_data.lst
#ls -1 ${DATA_DST_DIR}/train | grep "_label.npy" | sort > ${DATA_DST_DIR}/train/files_label.lst
#ls -1 ${DATA_DST_DIR}/validation | grep "_label.npy" | sort > ${DATA_DST_DIR}/validation/files_label.lst
