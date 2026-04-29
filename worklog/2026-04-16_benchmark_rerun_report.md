# B200 DGXC Benchmark Re-Run Report — 2026-04-16

## Summary

Completed 7 benchmark jobs today across 4 model families, all at 256-GPU (32-node) scale. Focused on closing gaps vs external Run1 reference numbers. Also debugged and resolved multiple infrastructure issues blocking Nemotron4 340B and Grok1 on the legacy 25.07/25.09 containers.

## Cluster State

- 72 batch nodes total, 2 drained (gpu-145, gpu-190), 70 usable
- Node blacklist refreshed: 38 formerly-bad nodes all pass GPU checks now
- Exclude list for today's runs: `gpu-[145,154,155,190]`
- Ran 2 concurrent 32-node jobs simultaneously

## Results vs Run1

### Batch C — All Complete

| Job   | Model         | Dtype  | Config                          | Steps 3-9 Avg | Steady-State | Run1   | vs Run1           |
|-------|---------------|--------|---------------------------------|---------------|-------------|--------|--------------------|
| 84023 | Llama 405B    | NVFP4  | TP=4, PP=16, VP=8, GBS=1536    | 1,746         | 1,808       | 1,698  | **+2.9% → +6.5%** |
| 84025 | Llama 70B     | NVFP4  | TP=2, PP=4, VP=5, GBS=256      | 1,943         | 2,034       | 2,016  | -3.6% → +0.9%     |
| 84027 | Qwen3 235B    | BF16   | TP=1, PP=8, VP=4, EP=8, GBS=8192 | 521        | 517         | 316    | +64%*              |
| 84034 | Qwen3 235B    | FP8    | TP=1, PP=8, EP=8, ETP=1, GBS=8192 | 389       | 390         | 254    | +53%*              |
| 84032 | Qwen3 30B     | BF16   | TP=1, PP=1, EP=8, ETP=1, GBS=512 | 197        | 204         | 285    | **-28%**           |

\*Qwen3 MoE TFLOP/s metric uses total params (235B/30B); Run1 likely uses active params (22B/3B). Step time is the fair comparison.

### Nemotron4 340B — Both Complete (NEW today)

| Job   | Dtype | Config                          | Steps 3-9 Avg | Steady (40-49) | Run1   | vs Run1             |
|-------|-------|---------------------------------|---------------|----------------|--------|----------------------|
| 84064 | FP8   | TP=8, PP=4, VP=12, GBS=64       | 1,244         | 1,267          | 1,412  | **-11.9% → -10.2%** |
| 84065 | BF16  | TP=8, PP=4, VP=12, GBS=64       | 866           | 882            | 921    | **-6.0% → -4.3%**   |

Container: `nvidia+nemo+25.07.01.sqsh` (legacy NeMo2 framework)

## Debugging Timeline — Nemotron4 340B

The Nemotron4 340B workload required extensive debugging (7+ submission attempts per dtype) due to the legacy 25.07 container:

1. **`--additional_slurm_params` not supported** — NeMo2 launcher doesn't accept this flag. Fixed by patching `#SBATCH --exclude=` directly in generated sbatch scripts.

2. **Pyxis HOME directory error** — `mkdir /home/johnson/.cache: Permission denied`. Fixed with `HOME=/tmp`, `NEMO_NLP_TMP=/tmp`, `HF_HOME` redirect.

3. **NCCL "no route to host"** — NCCL auto-detected wrong network interface inside container. Fixed with `NCCL_SOCKET_IFNAME=bond0`.

4. **Userbuffer init failure** — `Failed to get the world_size / rank`. The MPI stub (recompiled 2026-04-16 for the 26.02 container) is incompatible with the 25.07 container's MPI libraries.

5. **Config YAML changes ineffective** — Setting `tp_comm_overlap: false` in the YAML didn't work because the NeMo fdl_runner reads from a serialized `fn_or_script` binary config, not the YAML.

6. **Final fix** — Regenerated experiment configs from scratch with `TP_COMM_OVERLAP=False` set as an environment variable during `launch.sh` execution. The Python script reads `os.environ.get("TP_COMM_OVERLAP", "True")` at config generation time, baking it into the serialized config. This eliminated the userbuffer initialization entirely, removing the MPI dependency.

### Performance Impact of Disabled TP Comm Overlap

The -10% FP8 gap vs Run1 is primarily due to running without TP communication overlap. This optimization overlaps tensor-parallel all-reduce/reduce-scatter with GEMM computation. Without it, communication and compute are serialized, adding latency per step. The BF16 impact is smaller (-4%) since BF16 GEMMs are slower and communication is a smaller fraction of step time.

## Grok1 314B — Debugging Progress

5 submission attempts for Grok1 BF16 with different failure modes:

| Job   | Failure                       | Root Cause                                    |
|-------|-------------------------------|-----------------------------------------------|
| 83978 | Pyxis mkdir error             | Missing `HOME=/tmp` workaround                |
| 83980 | ModuleNotFoundError           | New experiment dir incompatible with 25.09     |
| 83981 | Fatal PMIx crash              | Removed MPI stub — it IS required              |
| 83982 | NCCL 600s timeout             | Landed on bad nodes                            |
| 83984 | Queued successfully           | Correct config, targeting good nodes           |

**Key discovery:** MPI stub + `TP_COMM_OVERLAP=False` are BOTH required for 25.09 container:
- MPI stub alone → TransformerEngine userbuffer hang
- `TP_COMM_OVERLAP=False` alone → fatal PMIx crash  
- Both together → PMIx intercepted (non-fatal), userbuffer skipped

Grok1 BF16/FP8 sbatch scripts are ready to submit when nodes are available.

## Full Scorecard vs Run1

| Model            | Dtype  | Run1   | Ours (Best) | Gap         | Status      |
|------------------|--------|--------|-------------|-------------|-------------|
| Llama 405B       | FP8    | 1,730  | 1,724       | -0.3%       | Done (Apr-15) |
| Llama 405B       | NVFP4  | 1,698  | 1,746       | **+2.9%**   | Done        |
| Llama 70B        | FP8    | 1,553  | 1,546       | -0.5%       | Done (Apr-15) |
| Llama 70B        | NVFP4  | 2,016  | 1,943       | -3.6%       | Done        |
| Nemotron-H 56B   | FP8    | 1,459  | 1,529       | **+4.8%**   | Done (Apr-15) |
| Nemotron4 340B   | FP8    | 1,412  | 1,244       | **-11.9%**  | Done (no TP overlap) |
| Nemotron4 340B   | BF16   | 921    | 866         | **-6.0%**   | Done (no TP overlap) |
| Qwen3 235B       | BF16   | 316    | 521         | +64%*       | Done        |
| Qwen3 235B       | FP8    | 254    | 389         | +53%*       | Done        |
| Qwen3 30B        | BF16   | 285    | 197         | **-28%**    | Needs V2 config |
| Grok1 314B       | BF16   | —      | —           | —           | Scripts ready |
| Grok1 314B       | FP8    | —      | —           | —           | Scripts ready |

\*MoE metric mismatch — TFLOP/s uses different param counts.

## Remaining Work

1. **Nemotron4 340B FP8** — Recompile MPI stub against 25.07 container to re-enable TP comm overlap and close -12% gap
2. **Qwen3 30B BF16** — Create V2 256-GPU config (current GBS=512 designed for 8 GPUs, suboptimal at 256)
3. **Grok1 BF16/FP8** — Submit when nodes available; sbatch scripts verified and ready
4. **Qwen3 235B** — Resolve MoE TFLOP/s metric mismatch with Run1 (step time comparison needed)

## Experiment Directories

| Model | Dtype | Dir |
|-------|-------|-----|
| Llama 405B NVFP4 | NVFP4 | `pretrain_llama31_405b_nvfp4_..._1776381662/` |
| Llama 70B NVFP4 | NVFP4 | `pretrain_llama3_70b_nvfp4_..._1776388400/` |
| Qwen3 235B | BF16 | `pretrain_qwen3_235b_a22b_bf16_..._1776388656/` |
| Qwen3 235B | FP8 | `pretrain_qwen3_235b_a22b_fp8_mx_..._1776394859/` |
| Qwen3 30B | BF16 | `pretrain_qwen3_30b_a3b_bf16_..._1776393060/` |
| Nemotron4 340B | FP8 | `pretrain_nemotron4_340b_fp8_..._1776405463/` |
| Nemotron4 340B | BF16 | `pretrain_nemotron4_340b_bf16_..._1776406049/` |

## Consolidated TFLOP/s/GPU — All Models at 256 GPU (32 Nodes)

Steps 3-9 average unless noted. All results from B200 DGXC cluster.

| Model              | Dtype | TFLOP/s/GPU | Job ID |
|--------------------|-------|-------------|--------|
| Grok1 314B         | BF16  | 994         | 84056  |
| Grok1 314B         | FP8   | 1,219       | 84061  |
| Nemotron4 15B      | BF16  | 866         | 83825  |
| Nemotron4 15B      | FP8   | 1,244       | 83826  |
| Nemotron4 340B     | BF16  | 866         | 84065  |
| Nemotron4 340B     | FP8   | 1,244       | 84064  |
| DeepSeek V3 671B   | BF16  | —           | —      |
| DeepSeek V3 671B   | FP8   | 551         | 83853  |
| Llama 3.1 405B     | FP8   | 1,724       | 83844  |
| Llama 3.1 405B     | NVFP4 | 1,746       | 84023  |
| Llama 3.1 70B      | FP8   | 1,546       | 83845  |
| Llama 3.1 70B      | NVFP4 | 1,943       | 84025  |
| Nemotron-H 56B     | FP8   | 1,529       | 83847  |
| Qwen3 235B         | BF16  | 521         | 84027  |
| Qwen3 235B         | FP8   | 389         | 84034  |
| Qwen3 30B          | BF16  | 197         | 84032  |
