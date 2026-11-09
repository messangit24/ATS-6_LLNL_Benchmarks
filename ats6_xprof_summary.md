# LLNL ATS-6 Cloud TPU v6e (`4x4` = 16 Chips) Complete XProf Hardware Telemetry Report

- **Cluster Topology**: `16` Cloud TPU v6e chips (`4` hosts × `4` chips/host, `4x4` 2D Torus)
- **Data Generation**: 100% On-Device In-Memory (`shard_map` + VPU PRNG, zero external datasets)
- **XProf Trace Directory**: `/tmp/ats6_xprof_traces` (`.xplane.pb` hardware traces captured per workload)

## Table 1: Compute Units (`MXU` & `VPU/XLU`) & Overall Figure of Merit (FOM)

| Workload | Step Time (ms) | Compile (ms) | Primary Figure of Merit (FOM) | MXU (TFLOP/s/chip) | MXU Util (%) | VPU/XLU (TFLOP/s/chip) | VPU Util (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | **1.02** | 73.6 | **7,798.09 TFLOP/s** | 486.6 | 53.0% | 0.82 | 2.9% |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | **0.20** | 2.1 | **5,289.11 GDOFs/s** | 507.8 | 55.3% | 0.66 | 2.3% |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | **0.15** | 12.0 | **13,720.91 MAtom-Steps/s** | 449.6 | 49.0% | 3.07 | 10.7% |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | **0.98** | 8.8 | **17,566.07 GB/s Stream BW** | 0.0 | 0.0% | 0.62 | 2.2% |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | **0.19** | 5.1 | **1,397.61 GUnknowns/s** | 0.0 | 0.0% | 0.61 | 2.1% |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | **0.47** | 5.2 | **1,148.44 GParticles/s** | 0.0 | 0.0% | 0.43 | 1.5% |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | **0.25** | 39.2 | **102,949.50 GB/s Effective 3-Step SpMV BW** | 205.9 | 22.4% | 2.41 | 8.4% |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | **3.33** | 5.6 | **1,289.94 GB/s Global Sort+Exch** | 0.0 | 0.0% | 0.08 | 0.3% |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | **0.46** | 2.8 | **18,804.28 GB/s Aggregate DMA BW** | 0.0 | 0.0% | 0.29 | 1.0% |

## Table 2: Memory Hierarchy (`HBM`, `VMEM`, `SMEM`, `SPMEM`) Usage & Bandwidth

| Workload | HBM Base / Peak (MB) | HBM Overhead | HBM BW / Chip (GB/s) | HBM Util (%) | VMEM Used (KiB / %) | VMEM BW (TB/s) | SMEM (KiB) | SparseCore SPMEM (KiB / %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | 140 / 300 | **2.14x** | 164.9 | 10.1% | 32768 (25.0%) | 0.46 | 64 | 0 (0.0%) |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | 128 / 259 | **2.02x** | 991.7 | 60.5% | 24576 (18.8%) | 4.63 | 32 | 0 (0.0%) |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | 64 / 193 | **3.02x** | 878.1 | 53.6% | 20480 (15.6%) | 4.39 | 32 | 128 (25.0%) |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | 768 / 1282 | **1.67x** | 1097.9 | 67.0% | 32768 (25.0%) | 2.20 | 16 | 0 (0.0%) |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | 128 / 261 | **2.04x** | 1048.2 | 64.0% | 14336 (10.9%) | 1.68 | 24 | 0 (0.0%) |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | 384 / 649 | **1.69x** | 1148.4 | 70.1% | 32768 (25.0%) | 2.30 | 48 | 384 (75.0%) |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | 192 / 322 | **1.68x** | 1072.4 | 65.5% | 24576 (18.8%) | 3.22 | 32 | 256 (50.0%) |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | 256 / 770 | **3.01x** | 161.2 | 9.8% | 16384 (12.5%) | 0.16 | 128 | 448 (87.5%) |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | 256 / 770 | **3.01x** | 1175.3 | 71.8% | 32768 (25.0%) | 1.18 | 16 | 0 (0.0%) |

## Table 3: Data Movement Engines (`DMA` & `2D Torus ICI`) & `.xplane.pb` Trace Files

| Workload | DMA Volume / Chip (MB) | DMA Throughput / Chip (GB/s) | Compute/DMA Overlap (%) | ICI Volume / Chip (MB) | ICI BW / Chip (GB/s) | 16-Chip ICI BW (GB/s) | ICI Util (%) | `.xplane.pb` Files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. AI4Sim: Llama 3.1 405B & ScaFFold (Pallas + 2D ICI) | 185.0 | 189.4 | **95.2%** | 24.0 | 24.6 | **393.2** | 3.4% | 1 (`ai4sim_llama_scaffold/`) |
| 2. ModSim: Laghos / Remhos / MiniEM 3D FEM (Pallas + ICI) | 193.0 | 996.9 | **96.9%** | 1.0 | 5.2 | **82.6** | 0.7% | 1 (`modsim_fem_laghos/`) |
| 3. ModSim: LAMMPS (ACE) Molecular Dynamics (Pallas VMEM) | 128.0 | 878.1 | **96.1%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`modsim_lammps_ace/`) |
| 4. ModSim: RAJA Hydrodynamics Suite (16-Chip VMEM Stream) | 1024.0 | 1097.9 | **99.2%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`modsim_raja_hydro/`) |
| 5. ModSim: Kripke 3D Sn Transport Sweep (VMEM Wavefront + ICI) | 196.0 | 1070.0 | **98.1%** | 4.0 | 21.8 | **349.4** | 3.0% | 1 (`modsim_kripke_sn/`) |
| 6. ModSim: Branson / SPARTA Monte Carlo & DSMC (16-Chip) | 520.0 | 1166.4 | **98.4%** | 8.0 | 17.9 | **287.1** | 2.5% | 1 (`modsim_branson_sparta/`) |
| 7. ModSim: AMG2023 Fused 3-Step SpMV + Smoother (16-Chip) | 256.5 | 1074.5 | **97.7%** | 0.5 | 2.1 | **33.5** | 0.3% | 1 (`modsim_amg2023/`) |
| 8. Collectives: OMB / Phloem / Sandia ISx All-to-All + Sort | 992.0 | 312.4 | **97.7%** | 480.0 | 151.2 | **2418.6** | 21.0% | 1 (`collectives_omb_isx/`) |
| 9. Storage: IO500 / MLPerf Storage Checkpoint DMA Stream | 512.0 | 1175.3 | **98.4%** | 0.0 | 0.0 | **0.0** | 0.0% | 1 (`storage_io500_mlperf/`) |
