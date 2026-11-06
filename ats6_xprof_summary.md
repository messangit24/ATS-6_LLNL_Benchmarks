# LLNL ATS-6 Cloud TPU v6e (`4x4` = 16 Chips) Complete XProf Hardware Telemetry Report

- **Cluster Topology**: `16` Cloud TPU v6e chips (`4` hosts × `4` chips/host, `4x4` 2D Torus)
- **Data Generation**: 100% On-Device In-Memory (`shard_map` + VPU PRNG, zero external datasets)
- **XProf Trace Directory**: `/tmp/ats6_xprof_traces` (`.xplane.pb` hardware traces captured per workload)

## Table 1: Compute Units (`MXU` & `VPU/XLU`) & Overall Figure of Merit (FOM)

| Workload | Step Time (ms) | Compile (ms) | Primary Figure of Merit (FOM) | MXU (TFLOP/s/chip) | MXU Util (%) | VPU/XLU (TFLOP/s/chip) | VPU Util (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | **1.64** | 15.5 | **4,888.13 TFLOP/s** | 304.6 | 33.2% | 0.92 | 3.2% |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | **0.40** | 2.9 | **333.69 GDOFs/s** | 32.0 | 3.5% | 0.04 | 0.1% |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | **0.42** | 2.7 | **2,497.68 MAtom-Steps/s** | 40.9 | 4.5% | 0.40 | 1.4% |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | **1.04** | 2.2 | **16,462.91 GB/s Stream BW** | 0.0 | 0.0% | 0.58 | 2.0% |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | **0.41** | 1.6 | **326.29 GUnknowns/s** | 0.0 | 0.0% | 0.14 | 0.5% |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | **0.69** | 6.5 | **779.29 GParticles/s** | 0.0 | 0.0% | 0.29 | 1.0% |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | **0.40** | 1.8 | **15,925.18 GB/s Effective 3-Step SpMV BW** | 31.9 | 3.5% | 0.37 | 1.3% |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | **19.44** | 126.5 | **55.23 GB/s Global Sort+Exch** | 0.0 | 0.0% | 0.18 | 0.6% |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | **0.70** | 2.2 | **12,188.63 GB/s Aggregate DMA BW** | 0.0 | 0.0% | 0.19 | 0.7% |

## Table 2: Memory Hierarchy (`HBM`, `VMEM`, `SMEM`, `SPMEM`) Usage & Bandwidth

| Workload | HBM Base / Peak (MB) | HBM Overhead | HBM BW / Chip (GB/s) | HBM Util (%) | VMEM Used (KiB / %) | VMEM BW (TB/s) | SMEM (KiB) | SparseCore SPMEM (KiB / %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | 140 / 285 | **2.03x** | 103.2 | 6.3% | 32768 (25.0%) | 0.29 | 64 | 0 (0.0%) |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | 48 / 113 | **2.36x** | 208.6 | 12.7% | 12288 (9.4%) | 0.67 | 32 | 0 (0.0%) |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | 64 / 193 | **3.02x** | 319.7 | 19.5% | 10240 (7.8%) | 0.96 | 32 | 128 (25.0%) |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | 768 / 1281 | **1.67x** | 1028.9 | 62.8% | 16384 (12.5%) | 2.06 | 16 | 0 (0.0%) |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | 64 / 128 | **2.01x** | 244.7 | 14.9% | 4096 (3.1%) | 0.39 | 24 | 0 (0.0%) |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | 384 / 641 | **1.67x** | 779.3 | 47.6% | 16384 (12.5%) | 1.56 | 48 | 384 (75.0%) |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | 96 / 161 | **1.68x** | 331.8 | 20.3% | 12288 (9.4%) | 1.00 | 32 | 256 (50.0%) |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | 64 / 197 | **3.09x** | 6.9 | 0.4% | 65536 (50.0%) | 0.07 | 128 | 448 (87.5%) |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | 256 / 770 | **3.01x** | 761.8 | 46.5% | 16384 (12.5%) | 0.76 | 16 | 0 (0.0%) |

## Table 3: Data Movement Engines (`DMA` & `2D Torus ICI`) & `.xplane.pb` Trace Files

| Workload | DMA Volume / Chip (MB) | DMA Throughput / Chip (GB/s) | Compute/DMA Overlap (%) | ICI Volume / Chip (MB) | ICI BW / Chip (GB/s) | 16-Chip ICI BW (GB/s) | ICI Util (%) | `.xplane.pb` Files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | 185.0 | 118.6 | **95.2%** | 24.0 | 15.4 | **246.2** | 2.1% | 1 (`ai4sim_llama_scaffold/`) |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | 80.5 | 209.9 | **96.2%** | 0.5 | 1.3 | **20.9** | 0.2% | 1 (`modsim_fem_laghos/`) |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | 128.0 | 319.7 | **98.0%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`modsim_lammps_ace/`) |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | 1024.0 | 1028.9 | **99.2%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`modsim_raja_hydro/`) |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | 97.0 | 247.3 | **98.9%** | 1.0 | 2.5 | **40.8** | 0.4% | 1 (`modsim_kripke_sn/`) |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | 516.0 | 785.4 | **99.2%** | 4.0 | 6.1 | **97.4** | 0.8% | 1 (`modsim_branson_sparta/`) |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | 128.5 | 333.1 | **97.6%** | 0.5 | 1.3 | **20.7** | 0.2% | 1 (`modsim_amg2023/`) |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | 248.0 | 13.4 | **91.6%** | 120.0 | 6.5 | **103.6** | 0.9% | 1 (`collectives_omb_isx/`) |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | 512.0 | 761.8 | **99.2%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`storage_io500_mlperf/`) |
