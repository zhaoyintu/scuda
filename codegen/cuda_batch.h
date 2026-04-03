#ifndef CUDA_BATCH_H
#define CUDA_BATCH_H

#include <cstddef>
#include <cstdint>

#define RPC_BATCH_CUDA_REGISTER 8000

// Sub-operation codes within a batch
#define CUDA_BATCH_FAT_BINARY     0
#define CUDA_BATCH_FUNCTION       1
#define CUDA_BATCH_VAR            2
#define CUDA_BATCH_FAT_BINARY_END 3

// Initialize the batch system
void cuda_batch_init();

// Check if batching is active (during registration phase)
int cuda_batch_is_active();

// Get the number of pending commands
uint32_t cuda_batch_pending_count();

// Append raw data to the batch buffer
void cuda_batch_append(uint8_t sub_op, const void *data, uint32_t data_size);

// Flush all buffered registration commands to the server as one RPC
// Returns 0 on success
int cuda_batch_flush();

// Flush only if there are pending commands, then disable batching
void cuda_batch_flush_if_pending();

// Flush pending batch but keep batching active (for FatBinary calls)
void cuda_batch_flush_and_continue();

// Mark that we're inside a FatBinary call (prevents flush_if_pending from disabling batch)
void cuda_batch_set_inside_fat_binary(int v);

// Look up a real fat binary handle by fake handle index
// Returns the real handle or nullptr if not found
void **cuda_batch_get_real_handle(uintptr_t fake_handle);

#endif
