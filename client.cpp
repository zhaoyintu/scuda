#include <arpa/inet.h>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <map>
#include <netdb.h>
#include <netinet/tcp.h>
#include <nvml.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <setjmp.h>
#include <signal.h>
#include <sys/mman.h>

#include "codegen/cuda_batch.h"
#include "codegen/gen_client.h"
#include "rpc.h"

pthread_mutex_t conn_mutex;
conn_t conns[16];
int nconns = 0;

const char *DEFAULT_PORT = "14833";

static int init = 0;
static jmp_buf catch_segfault;
static void *faulting_address = nullptr;

std::map<conn_t *, std::map<void *, size_t>> unified_devices;
std::map<void *, void *> host_funcs;

static void segfault(int sig, siginfo_t *info, void *unused) {
  faulting_address = info->si_addr;

  for (const auto &conn_entry : unified_devices) {
    for (const auto &[ptr, sz] : conn_entry.second) {
      if ((uintptr_t)ptr <= (uintptr_t)faulting_address &&
          (uintptr_t)faulting_address < ((uintptr_t)ptr + sz)) {
        // ensure we assign memory as close to the faulting address as
        // possible... by masking via the allocated unified memory size.
        uintptr_t aligned = (uintptr_t)faulting_address & ~(sz - 1);

        // Allocate memory at the faulting address
        void *allocated =
            mmap((void *)aligned, sz + (uintptr_t)faulting_address - aligned,
                 PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (allocated == MAP_FAILED) {
          perror("Failed to allocate memory at faulting address");
          _exit(1);
        }

        return;
      }
    }
  }

  // raise our original segfault handler
  struct sigaction sa;
  sa.sa_handler = SIG_DFL;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;

  if (sigaction(SIGSEGV, &sa, nullptr) == -1) {
    perror("Failed to reset SIGSEGV handler");
    _exit(EXIT_FAILURE);
  }

  raise(SIGSEGV);
}

int is_unified_pointer(conn_t *conn, void *arg) {
  auto conn_it = unified_devices.find(conn);
  if (conn_it == unified_devices.end()) {
    return 0;
  }

  // now check if the argument exists in the connection's mapped devices
  auto &devices = conn_it->second;
  if (devices.find(arg) != devices.end()) {
    return 1;
  }

  return 0;
}

int maybe_copy_unified_arg(conn_t *conn, void *arg, enum cudaMemcpyKind kind) {
  // find the connection in the map first
  auto conn_it = unified_devices.find(conn);
  if (conn_it == unified_devices.end()) {
    return 0;
  }

  // now find the argument in the sub-map for this connection
  auto &devices = conn_it->second;
  auto device_it = devices.find(arg);
  if (device_it != devices.end()) {
    std::cout << "Found unified arg pointer; copying..." << std::endl;

    void *ptr = device_it->first;
    size_t size = device_it->second;

    cudaError_t res = cudaMemcpy(ptr, ptr, size, kind);
    if (res != cudaSuccess) {
      std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(res)
                << std::endl;
      return -1;
    } else {
      std::cout << "Successfully copied " << size << " bytes" << std::endl;
    }
  }

  return 0;
}

static void set_segfault_handlers() {
  if (init > 0) {
    return;
  }

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = segfault;

  if (sigaction(SIGSEGV, &sa, NULL) == -1) {
    perror("sigaction");
    exit(EXIT_FAILURE);
  }

  init = 1;
}

void rpc_close(conn_t *conn) {
  if (pthread_mutex_lock(&conn_mutex) < 0)
    return;
  while (--nconns >= 0)
    close(conn->connfd);
  pthread_mutex_unlock(&conn_mutex);
}

typedef void (*func_t)(void *);

void add_host_node(void *fn, void *udata) { host_funcs[fn] = udata; }

void invoke_host_func(void *fn) {
  for (const auto &pair : host_funcs) {
    if (pair.first == fn) {
      func_t func = reinterpret_cast<func_t>(pair.first);
      std::cout << "Invoking function at: " << pair.first << std::endl;
      func(pair.second);
      return;
    }
  }
}

void *rpc_client_dispatch_thread(void *arg) {
  conn_t *conn = (conn_t *)arg;
  int op;

  while (true) {
    op = rpc_dispatch(conn, 1);

    if (op == 1) {
      std::cout << "Transferring memory..." << std::endl;

      int found = 0;

      rpc_read(conn, &found, sizeof(int));

      if (found > 0) {
        void *host_data = nullptr;
        void *dst = nullptr;
        const void *src = nullptr;
        size_t count = 0;
        cudaError_t result;
        int request_id;

        if (rpc_read(conn, &dst, sizeof(void *)) < 0 ||
            rpc_read(conn, &count, sizeof(size_t)) < 0) {
          std::cerr << "Failed to read transfer parameters." << std::endl;
          break;
        }

        host_data = malloc(count);
        if (!host_data) {
          std::cerr << "Memory allocation failed." << std::endl;
          break;
        }

        // Read the actual data from the server (sent from `src` in device
        // memory)
        if (rpc_read(conn, host_data, count) < 0) {
          std::cerr << "Failed to read device data from server." << std::endl;
          free(host_data);
          break;
        }

        // Copy received data to the destination (dst) on the host
        memcpy(dst, host_data, count);
      }

      void *temp_mem;
      if (rpc_read(conn, &temp_mem, sizeof(void *)) <= 0) {
        std::cerr << "rpc_read failed for mem. Closing connection."
                  << std::endl;
        break;
      }

      int request_id = rpc_read_end(conn);
      void *mem = temp_mem;

      if (mem == nullptr) {
        std::cerr << "Invalid function pointer!" << std::endl;
        continue;
      }

      invoke_host_func(mem);

      void *res = nullptr;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &res, sizeof(void *)) < 0 ||
          rpc_write_end(conn) < 0) {
        std::cerr << "rpc_write failed. Closing connection." << std::endl;
        break;
      }
    }
  }

  std::cerr << "Exiting dispatch thread due to an error." << std::endl;
  return nullptr;
}

int rpc_open() {
  // fprintf(stderr, "[SCUDA-DBG] rpc_open: entry\n");
  set_segfault_handlers();
  // fprintf(stderr, "[SCUDA-DBG] rpc_open: after set_segfault_handlers\n");

  sigsetjmp(catch_segfault, 1);
  // fprintf(stderr, "[SCUDA-DBG] rpc_open: after sigsetjmp\n");

  if (pthread_mutex_lock(&conn_mutex) < 0)
    return -1;
  // fprintf(stderr, "[SCUDA-DBG] rpc_open: after mutex lock\n");

  if (nconns > 0) {
    if (pthread_mutex_unlock(&conn_mutex) < 0)
      return -1;
    return 0;
  }

  // fprintf(stderr, "[SCUDA-DBG] rpc_open: getting SCUDA_SERVER env\n");

  char *server_ips = getenv("SCUDA_SERVER");
  if (server_ips == NULL) {
    fprintf(stderr, "SCUDA_SERVER environment variable not set\n");
    std::exit(1);
  }
  // fprintf(stderr, "[SCUDA-DBG] rpc_open: server=%s\n", server_ips);

  char *server_ip = strdup(server_ips);
  char *token;
  while ((token = strsep(&server_ip, ","))) {
    char *host;
    char *port;

    // Split the string into IP address and port
    char *colon = strchr(token, ':');
    if (colon == NULL) {
      host = token;
      port = const_cast<char *>(DEFAULT_PORT);
    } else {
      *colon = '\0';
      host = token;
      port = colon + 1;
    }

    // fprintf(stderr, "[SCUDA-DBG] rpc_open: resolving %s:%s\n", host, port);
    addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0) {
      // fprintf(stderr, "[SCUDA-DBG] getaddrinfo failed for %s:%s\n", host, port);
      continue;
    }
    // fprintf(stderr, "[SCUDA-DBG] rpc_open: getaddrinfo ok\n");

    int flag = 1;
    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd == -1) {
      // fprintf(stderr, "[SCUDA-DBG] socket creation failed\n");
      exit(1);
    }
    // fprintf(stderr, "[SCUDA-DBG] rpc_open: socket fd=%d\n", sockfd);

    int opts = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag,
                          sizeof(int));
    // fprintf(stderr, "[SCUDA-DBG] rpc_open: connecting to %s:%s...\n", host, port);
    if (connect(sockfd, res->ai_addr, res->ai_addrlen) < 0) {
      fprintf(stderr, "SCUDA: connect to %s:%s failed: %s\n", host, port, strerror(errno));
      exit(1);
    }
    // fprintf(stderr, "[SCUDA-DBG] rpc_open: connected!\n");

    conns[nconns] = {sockfd, 0};
    if (pthread_mutex_init(&conns[nconns].read_mutex, NULL) < 0 ||
        pthread_mutex_init(&conns[nconns].write_mutex, NULL) < 0) {
      return -1;
    }
    // fprintf(stderr, "[SCUDA-DBG] rpc_open: creating dispatch thread\n");

    pthread_create(&conns[nconns].read_thread, NULL, rpc_client_dispatch_thread,
                   (void *)&conns[nconns]);
    // fprintf(stderr, "[SCUDA-DBG] rpc_open: thread created, nconns=%d\n", nconns + 1);

    nconns++;
  }

  if (pthread_mutex_unlock(&conn_mutex) < 0)
    return -1;
  // fprintf(stderr, "[SCUDA-DBG] rpc_open: done, nconns=%d\n", nconns);
  if (nconns == 0)
    return -1;
  return 0;
}

conn_t *rpc_client_get_connection(unsigned int index) {
  cuda_batch_flush_if_pending();
  if (rpc_open() < 0)
    return nullptr;
  return &conns[index];
}

int rpc_size() { return nconns; }

void allocate_unified_mem_pointer(conn_t *conn, void *dev_ptr, size_t size) {
  // allocate new space for pointer mapping
  unified_devices[conn][dev_ptr] = size;
}

cudaError_t cuda_memcpy_unified_ptrs(conn_t *conn, cudaMemcpyKind kind) {
  auto conn_it = unified_devices.find(conn);
  if (conn_it == unified_devices.end()) {
    return cudaSuccess;
  }

  for (const auto &[ptr, sz] : conn_it->second) {
    size_t size = reinterpret_cast<size_t>(sz);

    // ptr is the same on both host/device
    cudaError_t res = cudaMemcpy(ptr, ptr, size, kind);
    if (res != cudaSuccess)
      return res;
  }
  return cudaSuccess;
}

void maybe_free_unified_mem(conn_t *conn, void *ptr) {
  auto conn_it = unified_devices.find(conn);
  if (conn_it == unified_devices.end()) {
    return;
  }

  for (const auto &[dev_ptr, sz] : conn_it->second) {
    size_t size = reinterpret_cast<size_t>(sz);

    if (dev_ptr == ptr) {
      munmap(dev_ptr, size);
      return;
    }
  }
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
  // cout removed

  auto it = get_function_pointer(symbol);
  if (it != nullptr) {
    *pfn = (void *)(&it);
    // cout removed
    return CUDA_SUCCESS;
  }

  if (strcmp(symbol, "cuGetProcAddress_v2") == 0 ||
      strcmp(symbol, "cuGetProcAddress") == 0) {
    *pfn = (void *)&cuGetProcAddress_v2;
    return CUDA_SUCCESS;
  }

  // cout removed

  // fall back to dlsym
  static void *(*real_dlsym)(void *, const char *) = NULL;
  if (real_dlsym == NULL) {
    real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym",
                                                         "GLIBC_2.2.5");
  }

  void *libCudaHandle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
  if (!libCudaHandle) {
    std::cerr << "Error: Failed to open libcuda.so" << std::endl;
    return CUDA_ERROR_UNKNOWN;
  }

  *pfn = real_dlsym(libCudaHandle, symbol);
  if (!(*pfn)) {
    std::cerr << "Error: Could not resolve symbol '" << symbol
              << "' using dlsym." << std::endl;
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}

// Stub for nvidia-smi internal version check
// nvidia-smi calls this to validate NVML library compatibility
typedef int nvmlReturn_t_stub;
static unsigned char dummy_export_table[256] = {0};
extern "C" nvmlReturn_t_stub nvmlInternalGetExportTable(const void **ppExportTable, const void *pExportTableId) {
  // fprintf(stderr, "[SCUDA-DBG] nvmlInternalGetExportTable called\n");
  if (ppExportTable)
    *ppExportTable = dummy_export_table;
  return 0; // NVML_SUCCESS
}

#define FAKE_NVML_HANDLE ((void *)0xDEAD0001)
#define FAKE_CUDA_HANDLE ((void *)0xDEAD0002)
#define FAKE_CUDART_HANDLE ((void *)0xDEAD0003)
#define FAKE_CUBLAS_HANDLE ((void *)0xDEAD0004)

void *dlopen(const char *filename, int flags) __THROW {
  static void *(*real_dlopen)(const char *, int) = NULL;
  if (!real_dlopen)
    real_dlopen = (void *(*)(const char *, int))dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.2.5");

  if (filename) {
    if (strstr(filename, "libnvidia-ml") || strstr(filename, "nvml"))
      return FAKE_NVML_HANDLE;
    if (strstr(filename, "libcuda.so"))
      return FAKE_CUDA_HANDLE;
    if (strstr(filename, "libcudart"))
      return FAKE_CUDART_HANDLE;
    if (strstr(filename, "libcublas"))
      return FAKE_CUBLAS_HANDLE;
  }
  return real_dlopen(filename, flags);
}

int dlclose(void *handle) __THROW {
  if (handle == FAKE_NVML_HANDLE || handle == FAKE_CUDA_HANDLE ||
      handle == FAKE_CUDART_HANDLE || handle == FAKE_CUBLAS_HANDLE)
    return 0;
  static int (*real_dlclose)(void *) = NULL;
  if (!real_dlclose)
    real_dlclose = (int (*)(void *))dlvsym(RTLD_NEXT, "dlclose", "GLIBC_2.2.5");
  return real_dlclose(handle);
}

void *dlsym(void *handle, const char *name) __THROW {
  // fprintf(stderr, "[SCUDA-DBG] dlsym: %s\n", name);

  void *func = get_function_pointer(name);

  /** proc address function calls are basically dlsym; we should handle this
   * differently at the top level. */
  if (strcmp(name, "cuGetProcAddress_v2") == 0 ||
      strcmp(name, "cuGetProcAddress") == 0) {
    return (void *)&cuGetProcAddress_v2;
  }

  if (func != nullptr) {
    return func;
  }

  // Handle internal NVIDIA functions not in the proxy
  if (strcmp(name, "nvmlInternalGetExportTable") == 0) {
    return (void *)&nvmlInternalGetExportTable;
  }

  // For fake handles, we can't call real_dlsym - return NULL for unknown funcs
  if (handle == FAKE_NVML_HANDLE || handle == FAKE_CUDA_HANDLE ||
      handle == FAKE_CUDART_HANDLE || handle == FAKE_CUBLAS_HANDLE) {
    // fprintf(stderr, "[SCUDA-DBG] dlsym: unknown func '%s' on fake handle\n", name);
    return NULL;
  }

  // Real dlsym lookup
  static void *(*real_dlsym)(void *, const char *) = NULL;
  if (real_dlsym == NULL) {
    real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym",
                                                         "GLIBC_2.2.5");
  }

  return real_dlsym(handle, name);
}
