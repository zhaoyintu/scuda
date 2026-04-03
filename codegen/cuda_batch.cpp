#include "cuda_batch.h"
#include "../rpc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// External RPC functions from client.cpp / rpc.cpp
extern conn_t *rpc_client_get_connection(unsigned int index);
extern int rpc_write_start_request(conn_t *conn, const int op);
extern int rpc_write(conn_t *conn, const void *data, const size_t size);
extern int rpc_wait_for_response(conn_t *conn);
extern int rpc_read(conn_t *conn, void *data, size_t size);
extern int rpc_read_end(conn_t *conn);

// Wire format:
// [cmd_count : uint32_t]
// For each command:
//   [sub_op : uint8_t]
//   [arg_size : uint32_t]
//   [arg_data : arg_size bytes]

static const size_t INITIAL_BUFFER_SIZE = 4 * 1024 * 1024;   // 4 MB
static const size_t MAX_BUFFER_SIZE     = 256 * 1024 * 1024; // 256 MB

// The batch buffer starts with a uint32_t cmd_count placeholder,
// followed by serialized commands.
static uint8_t *batch_buf      = nullptr;
static size_t   batch_buf_cap  = 0;
static size_t   batch_buf_used = 0; // bytes written (including the 4-byte header)
static uint32_t batch_cmd_count = 0;

// 1 while in registration phase (before first non-registration RPC)
static int batch_active = 1;

// Real handles returned by the server after flush, indexed by fat binary order
static std::vector<void **> real_handles;

void cuda_batch_init() {
    if (batch_buf != nullptr)
        return;

    batch_buf_cap  = INITIAL_BUFFER_SIZE;
    batch_buf      = (uint8_t *)malloc(batch_buf_cap);
    if (!batch_buf) {
        fprintf(stderr, "[cuda_batch] malloc failed\n");
        return;
    }
    // Reserve space for cmd_count at the beginning
    batch_buf_used = sizeof(uint32_t);
    batch_cmd_count = 0;
    batch_active = 1;
}

int cuda_batch_is_active() {
    // Initialize lazily on first call
    if (batch_buf == nullptr)
        cuda_batch_init();
    return batch_active;
}

uint32_t cuda_batch_pending_count() {
    return batch_cmd_count;
}

static int batch_ensure_capacity(size_t additional) {
    if (batch_buf == nullptr)
        cuda_batch_init();

    size_t needed = batch_buf_used + additional;
    if (needed <= batch_buf_cap)
        return 0;

    size_t new_cap = batch_buf_cap * 2;
    while (new_cap < needed)
        new_cap *= 2;
    if (new_cap > MAX_BUFFER_SIZE) {
        fprintf(stderr, "[cuda_batch] buffer would exceed max size (%zu MB)\n",
                MAX_BUFFER_SIZE / (1024 * 1024));
        return -1;
    }

    uint8_t *new_buf = (uint8_t *)realloc(batch_buf, new_cap);
    if (!new_buf) {
        fprintf(stderr, "[cuda_batch] realloc failed\n");
        return -1;
    }
    batch_buf     = new_buf;
    batch_buf_cap = new_cap;
    return 0;
}

void cuda_batch_append(uint8_t sub_op, const void *data, uint32_t data_size) {
    // Layout: [sub_op:u8][arg_size:u32][arg_data:data_size bytes]
    size_t needed = sizeof(uint8_t) + sizeof(uint32_t) + data_size;
    if (batch_ensure_capacity(needed) < 0)
        return;

    uint8_t *p = batch_buf + batch_buf_used;
    memcpy(p, &sub_op, sizeof(uint8_t));       p += sizeof(uint8_t);
    memcpy(p, &data_size, sizeof(uint32_t));   p += sizeof(uint32_t);
    if (data_size > 0)
        memcpy(p, data, data_size);

    batch_buf_used += needed;
    batch_cmd_count++;
}

int cuda_batch_flush() {
    if (batch_buf == nullptr || batch_cmd_count == 0)
        return 0;

    // Write cmd_count into the header slot
    memcpy(batch_buf, &batch_cmd_count, sizeof(uint32_t));

    // Temporarily mark inactive to prevent recursion from rpc_client_get_connection
    batch_active = 0;

    conn_t *conn = rpc_client_get_connection(0);
    if (!conn) {
        fprintf(stderr, "[cuda_batch] failed to get connection\n");
        return -1;
    }

    // fprintf(stderr, "[cuda_batch] flushing %u commands (%zu bytes)\n",
    //         batch_cmd_count, batch_buf_used);

    if (rpc_write_start_request(conn, RPC_BATCH_CUDA_REGISTER) < 0) {
        fprintf(stderr, "[cuda_batch] rpc_write_start_request failed\n");
        return -1;
    }

    if (rpc_write(conn, batch_buf, batch_buf_used) < 0) {
        fprintf(stderr, "[cuda_batch] rpc_write failed\n");
        return -1;
    }

    if (rpc_wait_for_response(conn) < 0) {
        fprintf(stderr, "[cuda_batch] rpc_wait_for_response failed\n");
        return -1;
    }

    // Response: [num_fat_binaries : uint32_t] followed by [handle : void*] each
    uint32_t num_handles = 0;
    if (rpc_read(conn, &num_handles, sizeof(uint32_t)) < 0) {
        fprintf(stderr, "[cuda_batch] rpc_read num_handles failed\n");
        return -1;
    }

    real_handles.resize(num_handles);
    for (uint32_t i = 0; i < num_handles; i++) {
        if (rpc_read(conn, &real_handles[i], sizeof(void **)) < 0) {
            fprintf(stderr, "[cuda_batch] rpc_read handle[%u] failed\n", i);
            return -1;
        }
    }

    if (rpc_read_end(conn) < 0) {
        fprintf(stderr, "[cuda_batch] rpc_read_end failed\n");
        return -1;
    }

    // fprintf(stderr, "[cuda_batch] flush complete: %u fat binaries registered\n",
    //         num_handles);

    // Reset buffer and re-enable batching for subsequent registration calls
    batch_buf_used  = sizeof(uint32_t);
    batch_cmd_count = 0;
    batch_active = 1;  // Stay active for more registration calls

    return 0;
}

// Track whether we're inside a FatBinary call to avoid disabling batching
static int inside_fat_binary = 0;

void cuda_batch_set_inside_fat_binary(int v) { inside_fat_binary = v; }

void cuda_batch_flush_if_pending() {
    if (batch_buf != nullptr && batch_active && batch_cmd_count > 0) {
        cuda_batch_flush();
    }
    // Only permanently disable batching if NOT called from FatBinary path
    if (!inside_fat_binary) {
        batch_active = 0;
    }
}

void cuda_batch_flush_and_continue() {
    // Flush pending batch but keep batching active (for FatBinary calls)
    if (batch_buf != nullptr && batch_cmd_count > 0) {
        cuda_batch_flush();
    }
    // batch_active remains 1 (set by cuda_batch_flush)
}

void **cuda_batch_get_real_handle(uintptr_t fake_handle) {
    uint32_t index = (uint32_t)(fake_handle & 0x0000FFFF);
    if (index < real_handles.size())
        return real_handles[index];
    return nullptr;
}
