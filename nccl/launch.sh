#!/bin/bash

#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%j.out
#SBATCH --comment=sysctl-sys.kernel.numa_balancing=0,transparent_hugepage_defrag=never,transparent_hugepage=never
#SBATCH --ntasks-per-node=8

# K8s DGX Cloud: NVIDIA_VISIBLE_DEVICES is set to GPU UUID strings by the outer K8s environment.
# Pyxis/enroot 98-nvidia.sh only reads it from --container-env, not from the inherited job env.
# Export it here so the sbatch script propagates it, and --container-env passes it into containers.
if [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
    export NVIDIA_VISIBLE_DEVICES
fi

function validate_running_environment() {
    :
    cd /nvcomms-perf-workspace
    # Check for libcudart.so and libnccl.so in common library paths
    cudart_found=0
    nccl_found=0
    
    for path in ${LD_LIBRARY_PATH//:/ } /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu; do
        if [ -d "$path" ] && [ -f "$path/libcudart.so" ]; then
            echo "Found libcudart.so in: $path"
            cudart_found=1
            break
        fi
    done
    for path in ${LD_LIBRARY_PATH//:/ } /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu; do
        if [ -d "$path" ] && [ -f "$path/libnccl.so" ]; then
            echo "Found libnccl.so in: $path"
            nccl_found=1
            break
        fi
    done

    # If both libraries are found in paths, return success
    if [ $cudart_found -eq 1 ] && [ $nccl_found -eq 1 ]; then
        return 0
    fi

    # Check ldconfig cache for missing libraries
    if [ $cudart_found -eq 0 ]; then
        cudart_cache=$(ldconfig -p | grep -oP 'libcudart\.so[^\s]*' | head -1)
        if [ -n "$cudart_cache" ]; then
            echo "Found libcudart.so in ldconfig: $cudart_cache"
            cudart_found=1
        fi
    fi
    
    if [ $nccl_found -eq 0 ]; then
        nccl_cache=$(ldconfig -p | grep -oP 'libnccl\.so[^\s]*' | head -1)
        if [ -n "$nccl_cache" ]; then
            echo "Found libnccl.so in ldconfig: $nccl_cache"
            nccl_found=1
        fi
    fi

    # Return success if both libraries are found (either in paths or ldconfig)
    if [ $cudart_found -eq 1 ] && [ $nccl_found -eq 1 ]; then
        return 0
    else
        [ $cudart_found -eq 0 ] && echo "Error: libcudart.so not found"
        [ $nccl_found -eq 0 ] && echo "Error: libnccl.so not found"
        return 1
    fi
}
export -f validate_running_environment

function get_nvl_domain_info() {
    cd /nvcomms-perf-workspace
    # Create a file per rank with NVLink domain information
    local rank_id=0
    if [ ! -z "$SLURM_PROCID" ]; then
        rank_id=$SLURM_PROCID
    elif [ ! -z "$OMPI_COMM_WORLD_RANK" ]; then
        rank_id=$OMPI_COMM_WORLD_RANK
    fi

    mkdir -p $LOG_DIR/nvl_domain_info
    nvidia-smi -q | grep -v GUID | grep -A4 Fabric > $LOG_DIR/nvl_domain_info/rank_${rank_id}.txt
}
export -f get_nvl_domain_info

function collect_sweep_metadata() {
    cd /nvcomms-perf-workspace
    . /etc/os-release
    hpcx_version=${HPCX_DIR##*/}

    while IFS=: read -r key value; do
        value=$(echo "$value" | xargs)
        [ "$value" = "null" -o -z "$value" ] && echo "$key: $value" || echo "$key: \"$value\""
    done <<EOF > $LOG_DIR/test_sweep_metadata.yml
name: ${TESTS:-null}
description: ${DESCRIPTION:-No description}
raw_data_path: ${LOG_DIR:-null}
data_source: ${USER:-$(whoami)}
data_reviewer: ${USER:-$(whoami)}
cluster_name: ${SLURM_CLUSTER_NAME:-null}
system_metadata:
  system_name: ${SYS_INFO_GPU_TYPE:-unknown}_${SYS_INFO_NETWORK:-unknown}_${SYS_INFO_SWITCH:-unknown}_${SYS_INFO_NIC:-unknown}_${SYS_INFO_NAME:-unknown}
  gpu_arch_type: $(nvidia-smi -q 2>/dev/null | grep "Product Name" | head -n1 | cut -d ":" -f2 || echo null)
  cpu_model_name: $(lscpu 2>/dev/null | grep -oP "(?<=Model name:).+$" || echo null)
  cpu_arch_type: $(lscpu 2>/dev/null | grep -oP "(?<=Architecture:).+$" || echo null)
  cuda_driver_version: $(nvidia-smi 2>/dev/null | grep -oP "(?<=Driver Version: )[\d\.]+" || echo null)
  mpi_type: $(mpirun --version 2>/dev/null | grep -i "open mpi" -q && echo openmpi || echo null)
  mpi_version: $(mpirun --version 2>/dev/null | grep -oP "(?<=\(Open MPI\) )[^\s]+$" || echo null)
  hpcx_version: ${hpcx_version:-null}
  os_type: ${ID:-null}
  os_version: ${VERSION:-null}
  linux_kernel_version: $(uname -r)
  adaptive_routing: null
network_metadata:
  nics: $(lspci 2>/dev/null | grep -E "Ethernet controller|Infiniband controller" | grep -v "BlueField" | awk -F": " '/Mellanox/ {found=1;print $NF;exit 0} END {if (!found) exit 1}' || echo null)
  switch_type: ${SYS_INFO_SWITCH:-null}
  switch_version: null
  network_name: ${SYS_INFO_NETWORK:-null}
  network_version: null
  network_plugin_name: null
  network_plugin_version: null
  mofed_version: $(command -v ofed_info >/dev/null && ofed_info -s 2>/dev/null | sed 's/:$//' || echo null)
  libfabric_version: $(command -v fi_info >/dev/null && fi_info --version 2>/dev/null | grep "Libfabric" | awk '{print $2}' || echo null)
EOF
}
export -f collect_sweep_metadata

function collect_test_set_metadata() {
    cd /nvcomms-perf-workspace
    for path in ${LD_LIBRARY_PATH//:/ } /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu; do
        [ -d "$path" ] && [ -f "$path/libnccl.so" ] && { libary_path="${path}/libnccl.so"; break; }
    done

    if [ -z "$libary_path" ]; then
        ldcache=$(ldconfig -p | awk -F' => ' '/libnccl\.so/ {print $2}' | head -1)
        [ -n "$ldcache" ] && { libary_path=$ldcache; }
    fi

    libary_version=$(strings $libary_path | grep -oP '(?<=NCCL version )[^ ]+(?= compiled with)')
    cuda_build_version=$(strings $libary_path | awk '/compiled with CUDA/ {print $NF}')


    while IFS=: read -r key value; do
        value=$(echo "$value" | xargs)
        [ "$value" = "null" -o -z "$value" ] && echo "$key: $value" || echo "$key: \"$value\""
    done <<EOF > $LOG_DIR/test_set_metadata.yml
timestamp: $(date +%s)
description: Allreduce, Allgather, ReduceScatter: NCCL Algo/Proto sweep. Alltoall, Sendrecv: NCCL_NCHANNELS_PER_NET_PEER sweep.
library:
  name: nccl
  path: ${libary_path:-null}
  version: ${libary_version:-null}
  commit_sha: ${library_commit_sha:-null}
  cuda_build_version: ${cuda_build_version:-null}
  cuda_runtime_version: $(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+(?=,)' || echo null)
env_variables:
  NCCL_DEBUG: WARN
  NCCL_NET: IB
EOF
}
export -f collect_test_set_metadata

function run_test() {
    cd /nvcomms-perf-workspace
    UNIQUE_NVL_DOMAINS=$(grep "ClusterUUID" $LOG_DIR/nvl_domain_info/rank_*.txt | awk '{print $NF}' | sort | uniq | wc -l || echo "null")
    GPUS_PER_NVL_DOMAIN=$(grep "ClusterUUID" $LOG_DIR/nvl_domain_info/rank_*.txt | awk '{print $NF}' | sort | uniq -c | sort -nr | awk '{print $1}' | paste -sd "," - || echo "null")

    test_iterations=1
    for iter in $(seq 1 ${test_iterations}); do
        log_name="${STEP_NAME}_iter${iter}"
        if [ "$SLURM_PROCID" -eq 0 ]; then
            while IFS=: read -r key value; do
            value=$(echo "$value" | xargs)
            [ "$value" = "null" -o -z "$value" ] && echo "$key: $value" || echo "$key: \"$value\""
        done <<EOF > $LOG_DIR/${log_name}.yml
test_name: ${TEST_NAME:-null}
log_name: ${log_name}.txt
timestamp: $(date +%s)
hostname: ${HOSTNAME:-$(hostname)}
container: $(echo "$TEST_CMD" | grep -oP -- '--container-image(?:=|\s+)\K\S+' || echo null)
num_nodes: ${SLURM_NNODES}
num_processes: ${SLURM_NTASKS}
num_process_per_node: ${SLURM_NTASKS_PER_NODE:-$((SLURM_NTASKS / (SLURM_NNODES ? SLURM_NNODES : 1)))}
unique_nvl_domains: ${UNIQUE_NVL_DOMAINS:-null}
num_gpus_per_nvl_domain: ${GPUS_PER_NVL_DOMAIN:-null}
slurm_id: ${SLURM_JOB_ID:-0}
slurm_job_name: ${SLURM_JOB_NAME:-null}
node_list: ${SLURM_JOB_NODELIST:-null}
test_binary: ${1##*/}
default_config: ${USE_DEFAULT_PARAM:-0}
iteration: $(((TESTSET_ITERATION - 1) * test_iterations + iter ))
cmd: "$TEST_CMD"
EOF
        fi

        log_file=${LOG_DIR}/${log_name}.txt
        rank_dir=${LOG_DIR}/${log_name}/rank${SLURM_PROCID}_$(hostname -s)
        rank_log_file=${rank_dir}/stdout.txt
        rank_err_file=${rank_dir}/stderr.txt

        mkdir -p $rank_dir

        # Override NCCL_DEBUG_FILE if it is set, substitute the path with the rank_dir
        if [ -n "$NCCL_DEBUG_FILE" ]; then
            export NCCL_DEBUG_FILE="${rank_dir}/${NCCL_DEBUG_FILE##*/}"
        fi

        # Don't use `cp ${rank_log_file} ${log_file}` on rank 0, because when cp is called,
        # file ${rank_log_file} might not exist yet due to disk cache or shared file system.
        if [ "$SLURM_PROCID" -eq 0 ]; then
            "$@" 2>${rank_err_file} | tee ${rank_log_file} | tee ${log_file}
        else
            "$@" 2>${rank_err_file} | tee ${rank_log_file}
        fi
        if [ $? -ne 0 ]; then
            echo "Error: Test ${TEST_NAME} failed, see ${rank_err_file} for details" >&2
            return $?
        fi
    done

    return 0
}
export -f run_test


WORKSPACE=${NVCOMMS_PERF_TOOLS_WORKSPACE:-$LLMB_INSTALL/workloads/microbenchmark_nccl/experiments/}
cd ${WORKSPACE}

export SWEEP_LOG_DIR="${SLURM_JOB_NAME}_container-nccl_${USER}_${SLURM_JOBID}"
export TESTSET_LOG_DIR="LOG_$(date "+%Y%m%d-%H%M")_${SLURM_JOBID}_h100_sweep_N${SLURM_JOB_NUM_NODES}"
export LOG_DIR="${SWEEP_LOG_DIR}"/"${TESTSET_LOG_DIR}"
mkdir -p "$LOG_DIR"

# System Environment Setup
export SYS_INFO_NAME="h100"
export SYS_INFO_NETWORK="IB"
export SYS_INFO_SWITCH="Quantum2"
export SYS_INFO_NIC="CX7"
export SYS_INFO_GPU_TYPE="h100-DGX"

# User Environment Setup
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/local/mpi"
export HPCX_HOME="/opt/hpcx/"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MPI_HOME/lib:$LD_LIBRARY_PATH"
export PATH="$MPI_HOME/bin:$PATH"
export SWEEP_LOG_ROOT="/nvcomms-workspace"
export NCCL_TEST_PATH="/usr/local/bin/"
export NVCOMMS_PERF_TOOLS_BINARY_SUFFIX="_mpi"
export NVCOMMS_PERF_TOOLS_WORKSPACE="$LLMB_INSTALL/workloads/microbenchmark_nccl/experiments/"


srun --overlap --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace bash -c "get_nvl_domain_info"
srun --overlap --ntasks=1 --nodes=1 --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace bash -c "validate_running_environment && collect_sweep_metadata && collect_test_set_metadata"
[ $? -ne 0 ] && { echo "Error: Failed to validate running environment"; exit 1; }

export MAX_GPUS_IN_NVL_DOMAIN=$(grep "ClusterUUID" $LOG_DIR/nvl_domain_info/rank_*.txt | awk '{print $NF}' | sort | uniq -c | sort -nr | awk '{print $1}' | head -n1 || echo "null")

testset_iterations=1

for testset_iter in $(seq 1 ${testset_iterations}); do
    echo "[${testset_iter}/${testset_iterations}] Running test 1/7"
    export TEST_NAME=all_reduce TESTSET_ITERATION=${testset_iter} USE_DEFAULT_PARAM=0
    export STEP_NAME="${TESTSET_ITERATION}_1_LOG_${TEST_NAME}_env1_args1_job${SLURM_JOB_ID}"
    export TEST_CMD=$(echo srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/all_reduce_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -dfloat -R0')

    mkdir -p "${LOG_DIR}"/"${STEP_NAME}"
    start_time=$(date +%s.%N)
    srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/all_reduce_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -dfloat -R0' > >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stdout.txt) 2> >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stderr.txt)
    ret_code=$?
    end_time=$(date +%s.%N)

    cat > "${LOG_DIR}"/"${STEP_NAME}"/result.txt << EOF
test_name="${TEST_NAME}"
test_command="${TEST_CMD}"
# Parameter combination string that will be used on this test run (envs | args | commands)
combination="env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" | -t1 -g1 -f2 -n20 -w5 -b8 -e16G -dfloat -R0"
ret_code="${ret_code}"
elapsed=$(echo "$end_time - $start_time" | bc)
EOF

    echo "[${testset_iter}/${testset_iterations}] Running test 2/7"
    export TEST_NAME=all_gather TESTSET_ITERATION=${testset_iter} USE_DEFAULT_PARAM=0
    export STEP_NAME="${TESTSET_ITERATION}_2_LOG_${TEST_NAME}_env1_args1_job${SLURM_JOB_ID}"
    export TEST_CMD=$(echo srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/all_gather_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -dfloat -R0')

    mkdir -p "${LOG_DIR}"/"${STEP_NAME}"
    start_time=$(date +%s.%N)
    srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/all_gather_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -dfloat -R0' > >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stdout.txt) 2> >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stderr.txt)
    ret_code=$?
    end_time=$(date +%s.%N)

    cat > "${LOG_DIR}"/"${STEP_NAME}"/result.txt << EOF
test_name="${TEST_NAME}"
test_command="${TEST_CMD}"
# Parameter combination string that will be used on this test run (envs | args | commands)
combination="env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" | -t1 -g1 -f2 -n20 -w5 -b8 -e16G -dfloat -R0"
ret_code="${ret_code}"
elapsed=$(echo "$end_time - $start_time" | bc)
EOF

    echo "[${testset_iter}/${testset_iterations}] Running test 3/7"
    export TEST_NAME=reduce_scatter TESTSET_ITERATION=${testset_iter} USE_DEFAULT_PARAM=0
    export STEP_NAME="${TESTSET_ITERATION}_3_LOG_${TEST_NAME}_env1_args1_job${SLURM_JOB_ID}"
    export TEST_CMD=$(echo srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/reduce_scatter_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -dfloat -R0')

    mkdir -p "${LOG_DIR}"/"${STEP_NAME}"
    start_time=$(date +%s.%N)
    srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/reduce_scatter_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -dfloat -R0' > >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stdout.txt) 2> >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stderr.txt)
    ret_code=$?
    end_time=$(date +%s.%N)

    cat > "${LOG_DIR}"/"${STEP_NAME}"/result.txt << EOF
test_name="${TEST_NAME}"
test_command="${TEST_CMD}"
# Parameter combination string that will be used on this test run (envs | args | commands)
combination="env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" | -t1 -g1 -f2 -n20 -w5 -b8 -e16G -dfloat -R0"
ret_code="${ret_code}"
elapsed=$(echo "$end_time - $start_time" | bc)
EOF

    echo "[${testset_iter}/${testset_iterations}] Running test 4/7"
    export TEST_NAME=alltoall TESTSET_ITERATION=${testset_iter} USE_DEFAULT_PARAM=0
    export STEP_NAME="${TESTSET_ITERATION}_4_LOG_${TEST_NAME}_env1_args1_job${SLURM_JOB_ID}"
    export TEST_CMD=$(echo srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="1" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/alltoall_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0')

    mkdir -p "${LOG_DIR}"/"${STEP_NAME}"
    start_time=$(date +%s.%N)
    srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="1" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/alltoall_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0' > >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stdout.txt) 2> >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stderr.txt)
    ret_code=$?
    end_time=$(date +%s.%N)

    cat > "${LOG_DIR}"/"${STEP_NAME}"/result.txt << EOF
test_name="${TEST_NAME}"
test_command="${TEST_CMD}"
# Parameter combination string that will be used on this test run (envs | args | commands)
combination="env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="1" | -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0"
ret_code="${ret_code}"
elapsed=$(echo "$end_time - $start_time" | bc)
EOF

    echo "[${testset_iter}/${testset_iterations}] Running test 5/7"
    export TEST_NAME=alltoall TESTSET_ITERATION=${testset_iter} USE_DEFAULT_PARAM=0
    export STEP_NAME="${TESTSET_ITERATION}_5_LOG_${TEST_NAME}_env2_args1_job${SLURM_JOB_ID}"
    export TEST_CMD=$(echo srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="2" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/alltoall_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0')

    mkdir -p "${LOG_DIR}"/"${STEP_NAME}"
    start_time=$(date +%s.%N)
    srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="2" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/alltoall_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0' > >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stdout.txt) 2> >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stderr.txt)
    ret_code=$?
    end_time=$(date +%s.%N)

    cat > "${LOG_DIR}"/"${STEP_NAME}"/result.txt << EOF
test_name="${TEST_NAME}"
test_command="${TEST_CMD}"
# Parameter combination string that will be used on this test run (envs | args | commands)
combination="env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="2" | -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0"
ret_code="${ret_code}"
elapsed=$(echo "$end_time - $start_time" | bc)
EOF

    echo "[${testset_iter}/${testset_iterations}] Running test 6/7"
    export TEST_NAME=sendrecv TESTSET_ITERATION=${testset_iter} USE_DEFAULT_PARAM=0
    export STEP_NAME="${TESTSET_ITERATION}_6_LOG_${TEST_NAME}_env1_args1_job${SLURM_JOB_ID}"
    export TEST_CMD=$(echo srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="2" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/sendrecv_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0')

    mkdir -p "${LOG_DIR}"/"${STEP_NAME}"
    start_time=$(date +%s.%N)
    srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="2" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/sendrecv_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0' > >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stdout.txt) 2> >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stderr.txt)
    ret_code=$?
    end_time=$(date +%s.%N)

    cat > "${LOG_DIR}"/"${STEP_NAME}"/result.txt << EOF
test_name="${TEST_NAME}"
test_command="${TEST_CMD}"
# Parameter combination string that will be used on this test run (envs | args | commands)
combination="env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="2" | -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0"
ret_code="${ret_code}"
elapsed=$(echo "$end_time - $start_time" | bc)
EOF

    echo "[${testset_iter}/${testset_iterations}] Running test 7/7"
    export TEST_NAME=sendrecv TESTSET_ITERATION=${testset_iter} USE_DEFAULT_PARAM=0
    export STEP_NAME="${TESTSET_ITERATION}_7_LOG_${TEST_NAME}_env2_args1_job${SLURM_JOB_ID}"
    export TEST_CMD=$(echo srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="4" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/sendrecv_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0')

    mkdir -p "${LOG_DIR}"/"${STEP_NAME}"
    start_time=$(date +%s.%N)
    srun --overlap --mpi=pmi2 -N ${SLURM_JOB_NUM_NODES} --cpu-bind=none --container-image=$LLMB_INSTALL/images/nvidia+nemo+25.09.00.sqsh --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES --container-mounts=./:/nvcomms-perf-workspace env NCCL_DEBUG=WARN env NCCL_NET=IB env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="4" env OMPI_MCA_pmix='^pmix3x' bash -c 'run_test ${NCCL_TEST_PATH}/sendrecv_perf${NVCOMMS_PERF_TOOLS_BINARY_SUFFIX} -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0' > >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stdout.txt) 2> >(tee "${LOG_DIR}"/"${STEP_NAME}"/job_step_stderr.txt)
    ret_code=$?
    end_time=$(date +%s.%N)

    cat > "${LOG_DIR}"/"${STEP_NAME}"/result.txt << EOF
test_name="${TEST_NAME}"
test_command="${TEST_CMD}"
# Parameter combination string that will be used on this test run (envs | args | commands)
combination="env NCCL_DEBUG="INFO" env NCCL_DEBUG_FILE="nccl_debug_file.%h.%p.txt" env NCCL_DEBUG_SUBSYS="Graph,Init,Env" env NCCL_NCHANNELS_PER_NET_PEER="4" | -t1 -g1 -f2 -n20 -w5 -b8 -e16G -duint8 -R0"
ret_code="${ret_code}"
elapsed=$(echo "$end_time - $start_time" | bc)
EOF

done

# Cleanup and completion
echo "All tests completed successfully"



# Archive the log directory
echo "Archiving log directory"
cp $0 ${SWEEP_LOG_DIR}/
tar -czf ${SWEEP_LOG_DIR}.tar.gz ${SWEEP_LOG_DIR}

echo "All done"

exit 0


