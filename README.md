# SCUDA: GPU-over-IP

SCUDA is a GPU over IP bridge allowing GPUs on remote machines to be attached to CPU-only machines. Any CUDA application — `nvidia-smi`, PyTorch, cuBLAS — can transparently use a remote GPU as if it were local.

## Quick Start

```bash
# On the GPU machine (server):
make server
./scuda_server

# On the CPU-only machine (client):
make local
./scuda.sh <server_ip> nvidia-smi
./scuda.sh <server_ip> python3 -c "import torch; print(torch.cuda.is_available())"
```

## How it works

SCUDA intercepts all CUDA/NVML API calls via `LD_PRELOAD` and forwards them as RPCs to a remote server that has a real GPU. The server executes the calls on the actual GPU and returns results.

```
┌─────────────────────┐         TCP/14833         ┌─────────────────────┐
│   Client (no GPU)   │ ◄══════════════════════► │  Server (has GPU)   │
│                     │                           │                     │
│  LD_PRELOAD hijacks │    RPC: cudaMalloc()      │  Calls real CUDA    │
│  CUDA/NVML calls    │ ─────────────────────►    │  runtime on GPU     │
│  and sends via RPC  │    Response: ptr, status   │  Returns results    │
│                     │ ◄─────────────────────    │                     │
└─────────────────────┘                           └─────────────────────┘
```

1372 CUDA/NVML/cuBLAS/cuDNN APIs are proxied, covering the full CUDA ecosystem.

## Performance

| Scenario | `import torch` time |
|----------|-------------------|
| Without SCUDA (local GPU) | ~3s |
| SCUDA (first run, cold cache) | ~3.5 min |
| SCUDA (subsequent runs, warm cache) | **~20s** |

The first run is slow because PyTorch registers ~10,000 CUDA kernels. SCUDA mitigates this with:
- **Batch registration**: Groups thousands of small registration RPCs into single network packets (18x fewer round-trips)
- **FatBinary caching**: Caches kernel registration data on disk (`~/.scuda/cache/`) so subsequent imports skip the multi-MB data transfer

## Server Setup (GPU Machine)

### Requirements

- NVIDIA GPU with working driver
- CUDA Toolkit 12.x (`nvcc`, headers, runtime libraries)
- cuDNN 8.x or 9.x
- cuBLAS (included with CUDA Toolkit)
- g++ with C++17 support
- Linux x86_64

### Install CUDA Toolkit

```bash
# Ubuntu/Debian - see https://developer.nvidia.com/cuda-downloads
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

# Verify
nvcc --version
nvidia-smi
```

### Install cuDNN

```bash
# See https://developer.nvidia.com/cudnn-downloads
sudo apt-get install -y libcudnn8 libcudnn8-dev
```

### Build & Run Server

```bash
cd scuda

# Build (uses /usr/local/cuda by default)
make server

# Or specify CUDA path:
make server CUDA_HOME=/usr/local/cuda-12.6

# Run
LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./scuda_server
# Output: "Server listening on port 14833..."

# Run as background daemon
LD_LIBRARY_PATH=/usr/local/cuda/lib64 nohup ./scuda_server > /tmp/scuda_server.log 2>&1 &
```

### One-Step Remote Deploy

From the client machine, you can deploy and build on the server in one command:

```bash
make deploy REMOTE=user@gpu-server
make start-remote REMOTE=user@gpu-server
```

### Custom Port

```bash
SCUDA_PORT=9999 ./scuda_server
```

## Client Setup (CPU-only Machine)

### Requirements

- Linux x86_64 (tested on WSL2, Ubuntu)
- g++ with C++17 support
- CUDA header files (headers only, no GPU/driver/runtime needed)
- Network connectivity to the server (port 14833)

### Install CUDA Headers (No GPU Needed)

You only need the header files for compilation. You can copy them from any machine with CUDA installed:

```bash
# Option 1: Copy from the server
mkdir -p /tmp/cuda_headers
scp -r user@gpu-server:/usr/local/cuda/include/* /tmp/cuda_headers/

# Option 2: Install CUDA Toolkit locally (only headers are used)
sudo apt-get install -y cuda-toolkit-12-6
# Then: make local CUDA_HEADERS=/usr/local/cuda/include
```

### Build Client

```bash
cd scuda

# Build (default headers path: /tmp/cuda_headers)
make local

# Or specify headers path:
make local CUDA_HEADERS=/usr/local/cuda/include

# This produces: libscuda_local.so
```

### Get nvidia-smi Binary

The client needs the `nvidia-smi` binary (it doesn't need to work locally — SCUDA intercepts its calls):

```bash
# Copy from server
scp user@gpu-server:/usr/bin/nvidia-smi /tmp/nvidia-smi
chmod +x /tmp/nvidia-smi
```

## Usage

### Using the wrapper script (recommended)

```bash
# nvidia-smi
./scuda.sh <server_ip> /tmp/nvidia-smi

# PyTorch
./scuda.sh <server_ip> python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
"

# Any CUDA program
./scuda.sh <server_ip> ./my_cuda_program
```

### Using environment variables

```bash
export SCUDA_SERVER=192.168.1.100
export LD_PRELOAD=/path/to/libscuda_local.so

nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Custom server port

```bash
./scuda.sh 192.168.1.100:9999 nvidia-smi
# or
export SCUDA_SERVER=192.168.1.100:9999
```

## Verified Working

| Application | Status |
|-------------|--------|
| `nvidia-smi` | Shows remote GPU name, driver version, power, utilization |
| `torch.cuda.is_available()` | Returns `True` |
| `torch.cuda.get_device_name(0)` | Returns remote GPU name (e.g., "NVIDIA GeForce RTX 4090") |
| `torch.cuda.device_count()` | Returns `1` |
| `cudaMalloc` / `cudaMemcpy` | Host↔Device data roundtrip verified correct |
| CUDA Driver API (`cuInit`, `cuDeviceGet`, etc.) | All return correct values |

## Architecture

```
scuda/
├── client.cpp              # Client entry: LD_PRELOAD hooks (dlopen/dlsym interception)
├── server.cpp              # Server entry: TCP listener, per-client threads
├── rpc.cpp / rpc.h         # RPC transport layer (TCP, scatter-gather I/O)
├── codegen/
│   ├── codegen.py          # Auto-generates gen_client.cpp / gen_server.cpp from CUDA headers
│   ├── gen_client.cpp      # Auto-generated: 1372 CUDA API client proxy functions
│   ├── gen_server.cpp      # Auto-generated: 1372 CUDA API server handlers
│   ├── manual_client.cpp   # Hand-written: __cudaRegister*, __cudaPush/PopCallConfiguration
│   ├── manual_server.cpp   # Hand-written: server handlers for registration + batch + cache
│   ├── gen_api.h           # RPC opcode definitions
│   ├── cuda_batch.cpp/h    # Batch registration: groups small RPCs into single packets
│   └── fatbin_cache.cpp/h  # FatBinary caching: skip re-sending kernel data on repeat imports
├── scuda.sh                # Wrapper script for easy usage
├── Makefile                # Build system (make local / make server)
└── gl/                     # OpenGL-over-IP proxy (experimental)
```

### Key Design Decisions

**LD_PRELOAD Interception**: The client `.so` overrides `dlopen()` and `dlsym()` to return fake handles for NVIDIA libraries and route function lookups to RPC proxy functions.

**Batch Registration**: PyTorch loads ~10,000 CUDA kernel registrations at import time. Instead of 10,000 network round-trips, SCUDA batches `__cudaRegisterVar`/`__cudaRegisterFunction` calls between `__cudaRegisterFatBinary` calls, reducing round-trips by 18x.

**FatBinary Caching**: Each `__cudaRegisterFatBinary` call transfers megabytes of kernel binary. SCUDA hashes the data and caches it — on repeat imports, only a 16-byte hash is sent instead of the full binary, and the server reuses its previously-registered handle. Client cache is stored at `~/.scuda/cache/`.

## Why RPC? Design Trade-offs

SCUDA uses a synchronous RPC (Remote Procedure Call) model: each CUDA API call becomes a network request-response. This is the simplest mapping — `cudaMalloc(size)` on the client becomes an RPC that the server executes on the real GPU and returns the result.

### The problem with per-call RPC

We experienced this firsthand: `import torch` generates ~10,000 kernel registration calls. With per-call RPC, each one is a network round-trip (~0.1ms on LAN). That's 10,000 × 0.1ms = **minutes of pure latency**, not computation.

```
Per-call RPC (naive):
  registerVar → wait → registerVar → wait → ... × 10,000  = slow

Batched (current):
  [registerVar × 35] → send batch → wait once              = 18x faster

Cached (current):
  hash check → hit → skip entirely                         = ~instant
```

### Better architectures (future directions)

| Approach | Idea | Benefit |
|----------|------|---------|
| **Command queue** | Accumulate calls, submit as a batch, get all results at once (like Vulkan/Metal) | Amortize round-trip cost across many calls |
| **One-way streaming** | Fire-and-forget for void calls (registrations, H2D memcpy), only sync when a return value is needed | 10,000 registrations in 1 round-trip |
| **RDMA** | Bypass TCP/IP, directly read/write remote GPU memory (InfiniBand/RoCE) | Latency from ~100μs to ~1μs |
| **Shared memory** | For same-machine VM/container scenarios, zero-copy via shared memory pages | Near-native performance |

Our batch registration and FatBinary caching are essentially retrofitting command-queue semantics onto the RPC framework. A ground-up redesign as an async command queue with selective synchronization would be the ideal long-term architecture.

## Troubleshooting

### `nvidia-smi` shows "ERR!" for temperature or "Function Not Found" for memory

Some nvidia-smi v2 internal functions (`nvmlDeviceGetTemperatureV`, `nvmlDeviceGetMemoryInfo_v2`) are not yet proxied. Core functionality works.

### `import torch` hangs

- Verify server is running: `ss -tlnp | grep 14833` on the server
- Check connectivity: `nc -zv <server_ip> 14833` from the client
- First import takes ~3-5 minutes (kernel registration). Subsequent imports use cache (~20s).

### Connection refused

- Check firewall: `sudo ufw allow 14833` on the server
- Check the server is listening: `ss -tlnp | grep 14833`

### Server crash / "Socket bind failed"

Port still in use from previous run. Kill old process:
```bash
pkill -9 scuda_server
sleep 2
./scuda_server
```

## Motivations

The goal of SCUDA is to enable developers to easily interact with GPUs over a network in order to take advantage of various pools of distributed GPUs.

### Use cases

1. **Local testing** - Verify CUDA compatibility from a CPU-only laptop using a remote GPU
2. **Aggregated GPU pools** - Centralize GPU management across multiple machines
3. **Remote model training** - Train models from low-power devices using remote GPUs
4. **Remote inferencing** - Run inference locally while offloading CUDA calls to remote GPUs
5. **Remote data processing** - Accelerate data operations on remote GPUs from local scripts

## Prior Art

- https://www.thundercompute.com/
- https://www.juicelabs.co/
- https://en.wikipedia.org/wiki/RCUDA (SCUDA's name: S is the next letter after R!)

## License

See [LICENSE](./LICENSE).
