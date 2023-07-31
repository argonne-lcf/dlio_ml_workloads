#!/bin/bash

usage() {
    echo "Usage: ${0} <NODES> <FRAMEWORK> <VERSION>"
    exit
}

# need three arguments:
[ "$#" -eq "3" ] || usage
# trick to test if a string is a valid integer
[ "${1}" -eq "${1}" ] || usage

NODES=${1}
[ "${NODES}" -ge "1" ] || usage

FRAMEWORK=${2}
VERSION=${3}

FRAMEWORK_LOWER=$(echo ${FRAMEWORK} | tr '[:upper:]' '[:lower:]')

# Set CamelCasing the way each framework's marketing likes:
FRAMEWORK=$(case "${FRAMEWORK_LOWER}" in
                "mxnet") echo "MxNet"; ;;
                "pytorch") echo "PyTorch"; ;;
                "tensorflow") echo "TensorFlow"; ;;
                * ) echo ${FRAMEWORK}; ;;
            esac)

if [[ "${NODES}" -gt "1" ]]; then
    SYSTEM_NAME="dgxa100_n${NODES}_ngc${VERSION}_${FRAMEWORK_LOWER}"
else
    # we special case our system name for single node:
    SYSTEM_NAME="dgxa100_ngc${VERSION}_${FRAMEWORK_LOWER}"
fi

replace@var() {
    sed -E "s/@@@${1}@@@/${!1}/g"
}

replace-all() {
    replace@var "SYSTEM_NAME" | replace@var "NODES" | replace@var "FRAMEWORK" | replace@var "VERSION"
}

cat <<EOF | replace-all  > ${SYSTEM_NAME}.json
{
    "submitter": "NVIDIA",
    "division": "closed",
    "status": "onprem",
    "system_name": "@@@SYSTEM_NAME@@@",
    "number_of_nodes": "@@@NODES@@@",
    "host_processors_per_node": "2",
    "host_processor_model_name": "AMD EPYC 7742",
    "host_processor_core_count": "64",
    "host_processor_vcpu_count": "",
    "host_processor_frequency": "",
    "host_processor_caches": "",
    "host_processor_interconnect": "",
    "host_memory_capacity": "2 TB",
    "host_storage_type": "NVMe SSD",
    "host_storage_capacity": "2x 1.92TB NVMe SSD + 30TB U.2 NVMe SSD",
    "host_networking": "Storage: 2x ConnectX-6 IB HDR 200Gb/sec, Management: 1x ConnectX-6 Ethernet 100Gb/Sec, Compute: 8x ConnectX-6 IB HDR 200Gb/Sec",
    "host_networking_topology": "",
    "host_memory_configuration": "",
    "accelerators_per_node": "8",
    "accelerator_model_name": "NVIDIA A100-SXM4-80GB (400W)",
    "accelerator_host_interconnect": "",
    "accelerator_frequency": "1.4GHz",
    "accelerator_on-chip_memories": "",
    "accelerator_memory_configuration": "HBM2e",
    "accelerator_memory_capacity": "80 GB",
    "accelerator_interconnect": "NVLINK 3.0, NVSWITCH 2.0 600GB/s",
    "accelerator_interconnect_topology": "",
    "cooling": "",
    "hw_notes": "",
    "framework": "@@@FRAMEWORK@@@ NVIDIA Release @@@VERSION@@@",
    "other_software_stack": {
        "cuda_version": "11.4.2.006",
        "cuda_driver_version": "470.57.02",
        "nccl_version": "2.11.4",
        "cublas_version": "11.6.1.51",
        "cudnn_version": "8.2.4.15",
        "trt_version": "8.0.3.0+cuda11.3.1.005",
        "dali_version": "1.5.0",
        "mofed_version": "5.4-rdmacore36.0",
        "openmpi_version": "4.1.1"
    },
    "operating_system": "Ubuntu 20.04.2 LTS",
    "sw_notes": ""
}
EOF
