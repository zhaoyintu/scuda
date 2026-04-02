#ifndef GL_BATCH_H
#define GL_BATCH_H

#include <stddef.h>
#include <stdint.h>

// conn_t is defined in rpc.h.  We include it here so that callers who only
// include batch.h still have a complete type for conn_t pointers.
#include "../rpc.h"

// ---------------------------------------------------------------------------
// Wire-protocol op codes
//
// These sit in a range well above any existing CUDA op codes so that a
// multiplexed connection could in principle carry both CUDA and GL traffic
// without ambiguity.
// ---------------------------------------------------------------------------

// OP_GL_BATCH  – one or more fire-and-forget GL commands bundled together.
//                Wire layout:
//                  [Count : uint32_t LE]
//                  repeated Count times:
//                    [FuncID  : uint16_t LE]
//                    [ArgSize : uint16_t LE]
//                    [ArgData : ArgSize bytes]
#define OP_GL_BATCH  9000

// OP_GL_SYNC   – flush any pending batch AND perform a synchronous round-trip
//                RPC (used for GL functions that return a value to the
//                client).  The server sends the actual response payload after
//                acknowledging the batch.
#define OP_GL_SYNC   9001

// OP_GL_FRAME  – signals the end of a rendered frame (SwapBuffers / Flush).
//                The server uses this to schedule presentation.
#define OP_GL_FRAME  9002

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// gl_batch_init must be called once before any other gl_batch_* function.
// It is safe to call from a static initialiser or __attribute__((constructor)).
void gl_batch_init(void);

// gl_get_connection returns the persistent conn_t used for all GL traffic.
// It lazily opens a TCP connection to SCUDA_SERVER on port SCUDA_PORT+1 the
// first time it is called.  Subsequent calls are effectively free (a single
// atomic load).  Returns NULL on failure.
conn_t *gl_get_connection(void);

// ---------------------------------------------------------------------------
// Batch construction
// ---------------------------------------------------------------------------

// gl_batch_append starts a new command record in the current batch.
// func_id must be the 16-bit GL function identifier (from gen_gl_api.h).
// Every call to gl_batch_append must be followed by zero or more calls to
// gl_batch_write and then exactly one call to either gl_batch_append (for the
// next command) or gl_batch_flush / gl_batch_flush_sync to finalise the batch.
void gl_batch_append(uint16_t func_id);

// gl_batch_write appends raw argument bytes to the command most recently
// started by gl_batch_append.  May be called multiple times per command.
// The combined size of all gl_batch_write calls for a single command must fit
// in a uint16_t (max 65535 bytes).
void gl_batch_write(const void *data, size_t size);

// ---------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------

// gl_batch_flush transmits all pending commands to the server as a single
// OP_GL_BATCH message and waits for the server's empty ACK response.
// The internal buffer is reset after a successful flush.
// Returns 0 on success, -1 on error.
int gl_batch_flush(void);

// gl_batch_flush_sync flushes any pending batch commands (same as
// gl_batch_flush) and then returns the connection so the caller can issue a
// synchronous RPC using the standard rpc_write / rpc_read helpers.
// Returns the connection on success, NULL on error.
conn_t *gl_batch_flush_sync(void);

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

// gl_batch_pending_count returns the number of commands currently waiting in
// the unsent batch.  Useful for deciding when to proactively flush.
int gl_batch_pending_count(void);

#endif // GL_BATCH_H
