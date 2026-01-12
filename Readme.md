# Performance Benchmarks

**Platform:** NVIDIA CUDA  
**Device:** NVIDIA GeForce GTX 1050

### Task: Super Boring Task

| Function | OpenMP | OpenCL Total | OpenCL Copy-In | OpenCL Kernel | OpenCL Copy-Out | OpenMP Bandwidth | OpenCL Bandwidth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| vector-times-vector | 8406us | 161761us | 111846us | 1992us | 47922us | 44.9069 GB/s | 189.502 GB/s |
| matrix-times-vector | 34536us | 586045us | 561102us | 24878us | 64us | 36.4413 GB/s | 50.5883 GB/s |
| matrix-times-matrix | 1736132us | 29166us | 14531us | 9539us | 5095us | 0.021743 GB/s | 3.95731 GB/s |

### Task: Reduce and Scan

| Function | OpenMP | OpenCL Total | OpenCL Copy-In | OpenCL Kernel | OpenCL Copy-Out | OpenMP Bandwidth | OpenCL Bandwidth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| reduce | 3167us | 63346us | 58737us | 2461us | 2148us | 79.4627 GB/s | 102.259 GB/s |
| scan-inclusive | 9451us | 74429us | 55753us | 4546us | 14130us | 13.3138 GB/s | 27.6791 GB/s |

### Task: Filter

| Function | OpenMP | OpenCL Total | OpenCL Copy-In | OpenCL Kernel | OpenCL Copy-Out |
| :--- | :--- | :--- | :--- | :--- | :--- |
| filter | 6614us | 10156us | 6551us | 1299us | 2305us |