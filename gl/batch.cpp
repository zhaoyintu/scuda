// gl/batch.cpp – OpenGL command batching engine for SCUDA's GL-over-IP proxy.
//
// Design overview
// ---------------
// OpenGL workloads consist of hundreds of small calls per frame (state
// changes, draw calls, uniform uploads, …).  Forwarding each call as its own
// RPC round-trip would saturate the network with tiny messages.  Instead we
// accumulate commands into a linear byte buffer and flush them in bulk:
//
//   client side                          server side
//   -----------                          -----------
//   gl_batch_append(FUNC_glUniform4f)
//   gl_batch_write(&v, 16)
//   gl_batch_append(FUNC_glDrawArrays)
//   gl_batch_write(&mode, 4)
//   …
//   gl_batch_flush()   ──── OP_GL_BATCH ──────────► dispatch_gl_batch()
//                      ◄─── empty ACK ─────────────
//
// For GL functions that return a value the caller uses gl_batch_flush_sync()
// to drain the pending batch first and then performs a normal synchronous RPC
// on the same connection.
//
// Wire format
// -----------
// Each OP_GL_BATCH payload is:
//
//   [Count   : uint32_t, little-endian]          – number of commands
//   repeated Count times:
//     [FuncID  : uint16_t, little-endian]        – GL API function id
//     [ArgSize : uint16_t, little-endian]        – byte length of ArgData
//     [ArgData : ArgSize bytes]                  – packed arguments
//
// All integers are written in native byte order (x86 LE in practice); a
// production implementation would hton/ntoh at the boundary.

#include "batch.h"
#include "../rpc.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netdb.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Compile-time tunables
// ---------------------------------------------------------------------------

// Initial capacity of the batch payload buffer (bytes).
static constexpr size_t BATCH_INITIAL_CAPACITY = 64u * 1024u;   // 64 KB

// Maximum capacity; grow() will refuse to exceed this.
static constexpr size_t BATCH_MAX_CAPACITY     = 1u * 1024u * 1024u; // 1 MB

// Default SCUDA port (mirrors the value in client.cpp).
static constexpr uint16_t SCUDA_DEFAULT_PORT = 14833;

// GL proxy listens on SCUDA_PORT + 1.
static constexpr uint16_t GL_PORT_OFFSET = 1;

// ---------------------------------------------------------------------------
// Batch buffer
// ---------------------------------------------------------------------------

// The batch buffer stores the raw wire payload that will be sent as the body
// of a single OP_GL_BATCH RPC message.
//
// Layout while being built:
//   bytes [0..3]          – reserved for the command count (filled at flush)
//   bytes [4..]           – command records appended by gl_batch_append /
//                           gl_batch_write
//
// Within a command record the ArgSize field is written lazily: we record the
// offset of the ArgSize field when the command starts (arg_size_offset) and
// then fill it in just before writing the next command header or flushing.

struct BatchBuffer {
    uint8_t *data;          // heap-allocated payload buffer
    size_t   capacity;      // allocated bytes
    size_t   size;          // bytes in use (always >= 4 after init)

    // Number of complete commands in the buffer.  The count field at bytes
    // [0..3] is written at flush time from this value.
    uint32_t cmd_count;

    // Byte offset of the ArgSize field of the command currently being built.
    // A value of 0 means no command is open (0 is the count field, so it can
    // never be a valid arg_size_offset).
    size_t arg_size_offset;
};

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

// Single batch buffer for the GL rendering thread.  GL is inherently
// single-threaded (one context per thread) but we protect the state with a
// mutex so that diagnostic calls from other threads (e.g. gl_batch_pending_count)
// are safe.
static BatchBuffer     s_batch;
static pthread_mutex_t s_batch_mutex  = PTHREAD_MUTEX_INITIALIZER;

// The persistent connection to the GL server.
static conn_t          s_gl_conn;
static pthread_once_t  s_conn_once    = PTHREAD_ONCE_INIT;
static int             s_conn_valid   = 0; // 1 when s_gl_conn is usable

// ---------------------------------------------------------------------------
// Internal helpers – buffer management
// ---------------------------------------------------------------------------

// Reserve at least `need` additional bytes in the buffer, doubling capacity
// until the requirement is met.  Returns 0 on success, -1 on failure.
static int batch_grow(size_t need) {
    size_t required = s_batch.size + need;
    if (required <= s_batch.capacity)
        return 0; // already enough room

    size_t new_cap = s_batch.capacity;
    while (new_cap < required) {
        if (new_cap >= BATCH_MAX_CAPACITY) {
            fprintf(stderr, "[gl_batch] buffer capacity limit (%zu bytes) reached\n",
                    BATCH_MAX_CAPACITY);
            return -1;
        }
        new_cap = (new_cap * 2 <= BATCH_MAX_CAPACITY) ? new_cap * 2
                                                       : BATCH_MAX_CAPACITY;
    }

    uint8_t *p = static_cast<uint8_t *>(realloc(s_batch.data, new_cap));
    if (!p) {
        fprintf(stderr, "[gl_batch] realloc failed: %s\n", strerror(errno));
        return -1;
    }
    s_batch.data     = p;
    s_batch.capacity = new_cap;
    return 0;
}

// Write `size` bytes from `src` at the current tail of the buffer.
// Grows the buffer if needed.  Returns 0 on success, -1 on failure.
static int batch_raw_write(const void *src, size_t size) {
    if (batch_grow(size) < 0)
        return -1;
    memcpy(s_batch.data + s_batch.size, src, size);
    s_batch.size += size;
    return 0;
}

// Overwrite `size` bytes at absolute offset `off` without changing s_batch.size.
static void batch_patch(size_t off, const void *src, size_t size) {
    memcpy(s_batch.data + off, src, size);
}

// ---------------------------------------------------------------------------
// Internal helpers – command finalisation
// ---------------------------------------------------------------------------

// If a command is currently open, compute its argument byte count and write it
// into the reserved ArgSize field.  Marks the command as closed afterwards.
static void finalize_open_command(void) {
    if (s_batch.arg_size_offset == 0)
        return; // no open command

    // ArgData starts right after the ArgSize field (2 bytes).
    size_t arg_data_start = s_batch.arg_size_offset + sizeof(uint16_t);
    size_t arg_data_size  = s_batch.size - arg_data_start;

    if (arg_data_size > UINT16_MAX) {
        fprintf(stderr, "[gl_batch] single command argument too large (%zu bytes)\n",
                arg_data_size);
        // Truncate to keep the wire format valid; the server will get garbage
        // but at least we don't corrupt the framing.
        arg_data_size = UINT16_MAX;
    }

    uint16_t arg_size16 = static_cast<uint16_t>(arg_data_size);
    batch_patch(s_batch.arg_size_offset, &arg_size16, sizeof(arg_size16));

    s_batch.arg_size_offset = 0; // mark closed
    s_batch.cmd_count++;
}

// ---------------------------------------------------------------------------
// Internal helpers – connection management
// ---------------------------------------------------------------------------

// Called exactly once via pthread_once to establish the GL TCP connection.
static void connect_once(void) {
    const char *server_env = getenv("SCUDA_SERVER");
    if (!server_env) {
        fprintf(stderr, "[gl_batch] SCUDA_SERVER not set; GL forwarding disabled\n");
        return;
    }

    // Parse "host" or "host:port" from the first comma-separated token.
    // We deliberately only use the first server entry; load-balancing across
    // multiple GL servers is not yet implemented.
    char buf[256];
    strncpy(buf, server_env, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    // Strip everything from the first comma onward.
    char *comma = strchr(buf, ',');
    if (comma) *comma = '\0';

    char *host = buf;
    char  port_str[16];

    char *colon = strchr(buf, ':');
    uint16_t base_port = SCUDA_DEFAULT_PORT;
    if (colon) {
        *colon = '\0';
        base_port = static_cast<uint16_t>(atoi(colon + 1));
    }
    snprintf(port_str, sizeof(port_str), "%u",
             static_cast<unsigned>(base_port + GL_PORT_OFFSET));

    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, port_str, &hints, &res) != 0) {
        fprintf(stderr, "[gl_batch] getaddrinfo(%s:%s) failed: %s\n",
                host, port_str, strerror(errno));
        return;
    }

    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd < 0) {
        fprintf(stderr, "[gl_batch] socket() failed: %s\n", strerror(errno));
        freeaddrinfo(res);
        return;
    }

    // Disable Nagle – we control batching ourselves.
    int flag = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<const char *>(&flag), sizeof(flag));

    if (connect(sockfd, res->ai_addr, res->ai_addrlen) < 0) {
        fprintf(stderr, "[gl_batch] connect(%s:%s) failed: %s\n",
                host, port_str, strerror(errno));
        close(sockfd);
        freeaddrinfo(res);
        return;
    }
    freeaddrinfo(res);

    // Initialise the conn_t.  We zero the whole struct first to ensure any
    // padding / future fields are clean.
    memset(&s_gl_conn, 0, sizeof(s_gl_conn));
    s_gl_conn.connfd = sockfd;

    if (pthread_mutex_init(&s_gl_conn.read_mutex,  nullptr) != 0 ||
        pthread_mutex_init(&s_gl_conn.write_mutex, nullptr) != 0 ||
        pthread_cond_init (&s_gl_conn.read_cond,   nullptr) != 0) {
        fprintf(stderr, "[gl_batch] pthread init failed: %s\n", strerror(errno));
        close(sockfd);
        return;
    }

    s_conn_valid = 1;
    fprintf(stderr, "[gl_batch] connected to GL server at %s:%s\n", host, port_str);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void gl_batch_init(void) {
    // Allocate the initial batch payload buffer.
    s_batch.data     = static_cast<uint8_t *>(malloc(BATCH_INITIAL_CAPACITY));
    s_batch.capacity = BATCH_INITIAL_CAPACITY;

    if (!s_batch.data) {
        fprintf(stderr, "[gl_batch] initial malloc failed\n");
        abort();
    }

    // Reserve 4 bytes at the front for the command count (written at flush).
    memset(s_batch.data, 0, sizeof(uint32_t));
    s_batch.size           = sizeof(uint32_t);
    s_batch.cmd_count      = 0;
    s_batch.arg_size_offset = 0;
}

conn_t *gl_get_connection(void) {
    pthread_once(&s_conn_once, connect_once);
    return s_conn_valid ? &s_gl_conn : nullptr;
}

void gl_batch_append(uint16_t func_id) {
    pthread_mutex_lock(&s_batch_mutex);

    // Close any previously open command before starting a new one.
    finalize_open_command();

    // If the buffer hasn't been initialised yet (gl_batch_init was not called),
    // do a best-effort initialisation now.  This should not happen in normal
    // usage, but defensive code avoids a hard crash.
    if (!s_batch.data) {
        pthread_mutex_unlock(&s_batch_mutex);
        gl_batch_init();
        pthread_mutex_lock(&s_batch_mutex);
    }

    // Write the FuncID field.
    batch_raw_write(&func_id, sizeof(func_id));

    // Reserve the ArgSize field; record where it is so we can fill it later.
    s_batch.arg_size_offset = s_batch.size;
    uint16_t placeholder    = 0;
    batch_raw_write(&placeholder, sizeof(placeholder));

    // ArgData will be appended by subsequent gl_batch_write calls.

    pthread_mutex_unlock(&s_batch_mutex);
}

void gl_batch_write(const void *data, size_t size) {
    if (!data || size == 0)
        return;

    pthread_mutex_lock(&s_batch_mutex);

    if (s_batch.arg_size_offset == 0) {
        // No command is open – this is a programming error.
        fprintf(stderr, "[gl_batch] gl_batch_write called without gl_batch_append\n");
        pthread_mutex_unlock(&s_batch_mutex);
        return;
    }

    batch_raw_write(data, size);

    pthread_mutex_unlock(&s_batch_mutex);
}

int gl_batch_flush(void) {
    pthread_mutex_lock(&s_batch_mutex);

    // Nothing to send – return success immediately.
    if (s_batch.cmd_count == 0 && s_batch.arg_size_offset == 0) {
        pthread_mutex_unlock(&s_batch_mutex);
        return 0;
    }

    // Finalise any open command and stamp the count into the header.
    finalize_open_command();

    // Sanity-check: if we still have no commands something went wrong.
    if (s_batch.cmd_count == 0) {
        pthread_mutex_unlock(&s_batch_mutex);
        return 0;
    }

    batch_patch(0, &s_batch.cmd_count, sizeof(s_batch.cmd_count));

    conn_t *conn = gl_get_connection();
    if (!conn) {
        fprintf(stderr, "[gl_batch] no GL connection – dropping %u commands\n",
                s_batch.cmd_count);
        // Reset so the client can continue without crashing.
        s_batch.size      = sizeof(uint32_t);
        s_batch.cmd_count = 0;
        memset(s_batch.data, 0, sizeof(uint32_t));
        pthread_mutex_unlock(&s_batch_mutex);
        return -1;
    }

    // Capture the data we need to send before releasing the batch mutex.
    // rpc_write_start_request will take the connection's write_mutex, which is
    // a different lock, so we must not hold both simultaneously.
    size_t   payload_size = s_batch.size;
    uint8_t *payload_copy = static_cast<uint8_t *>(malloc(payload_size));
    if (!payload_copy) {
        fprintf(stderr, "[gl_batch] malloc for flush copy failed\n");
        pthread_mutex_unlock(&s_batch_mutex);
        return -1;
    }
    memcpy(payload_copy, s_batch.data, payload_size);

    // Reset the batch buffer (keep the header space reserved).
    s_batch.size           = sizeof(uint32_t);
    s_batch.cmd_count      = 0;
    s_batch.arg_size_offset = 0;
    memset(s_batch.data, 0, sizeof(uint32_t));

    pthread_mutex_unlock(&s_batch_mutex);

    // --- Send via SCUDA RPC ---------------------------------------------------
    //
    // rpc_write_start_request locks conn->write_mutex and sets up the iovec
    // scatter-gather list.  rpc_write appends our payload pointer (zero-copy).
    // rpc_write_end issues writev() and releases the write lock, returning the
    // request id that the server will echo back in its ACK.
    //
    // The server is expected to send an empty (zero-payload) response for every
    // OP_GL_BATCH so that the client knows the commands have been consumed and
    // the server is ready for the next batch.  We read it here to provide
    // back-pressure and prevent unbounded pipeline depth.

    if (rpc_write_start_request(conn, OP_GL_BATCH) < 0) {
        fprintf(stderr, "[gl_batch] rpc_write_start_request failed\n");
        free(payload_copy);
        return -1;
    }

    if (rpc_write(conn, payload_copy, payload_size) < 0) {
        fprintf(stderr, "[gl_batch] rpc_write failed\n");
        free(payload_copy);
        // write_mutex is still held; release it by calling rpc_write_end and
        // ignoring the result.
        rpc_write_end(conn);
        return -1;
    }

    int write_id = rpc_write_end(conn);
    free(payload_copy); // safe: writev has already copied into kernel buffers
    if (write_id < 0) {
        fprintf(stderr, "[gl_batch] rpc_write_end failed\n");
        return -1;
    }

    // Wait for the server's ACK (empty response keyed to our write_id).
    if (rpc_read_start(conn, write_id) < 0) {
        fprintf(stderr, "[gl_batch] rpc_read_start failed waiting for ACK\n");
        return -1;
    }
    // The ACK carries no payload; just release the read lock.
    if (rpc_read_end(conn) < 0) {
        fprintf(stderr, "[gl_batch] rpc_read_end failed\n");
        return -1;
    }

    return 0;
}

conn_t *gl_batch_flush_sync(void) {
    // Drain any pending batch so the server processes all queued commands
    // before we issue the synchronous call.  A sync call typically reads a
    // return value, so ordering matters.
    if (gl_batch_pending_count() > 0) {
        if (gl_batch_flush() < 0)
            return nullptr;
    }

    return gl_get_connection();
}

int gl_batch_pending_count(void) {
    pthread_mutex_lock(&s_batch_mutex);

    // An open (not-yet-finalised) command counts as pending.
    int count = static_cast<int>(s_batch.cmd_count);
    if (s_batch.arg_size_offset != 0)
        count += 1; // one open command not yet counted in cmd_count

    pthread_mutex_unlock(&s_batch_mutex);
    return count;
}
