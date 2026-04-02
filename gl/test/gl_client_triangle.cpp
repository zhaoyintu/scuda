/**
 * GL-over-IP Client — Triangle Test
 *
 * Connects to remote GL server, sends commands to render a colored triangle,
 * reads back pixels, and verifies the result.
 *
 * Build: g++ -o gl_client gl_client_triangle.cpp -lpthread
 * Run:   ./gl_client <server_ip> [port]
 */

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netdb.h>
#include <vector>
#include <string>
#include <cstdint>

// ── Protocol (must match server) ────────────────────────────────────────────
#define MSG_GL_BATCH  1
#define MSG_GL_SYNC   2
#define MSG_GL_FRAME  3
#define MSG_GL_ACK    4

// GL constants (from GL headers)
#define MY_GL_COLOR_BUFFER_BIT   0x00004000
#define MY_GL_DEPTH_BUFFER_BIT   0x00000100
#define MY_GL_TRIANGLES          0x0004
#define MY_GL_FLOAT              0x1406
#define MY_GL_UNSIGNED_BYTE      0x1401
#define MY_GL_FALSE              0
#define MY_GL_TRUE               1
#define MY_GL_ARRAY_BUFFER       0x8892
#define MY_GL_STATIC_DRAW        0x88E4
#define MY_GL_VERTEX_SHADER      0x8B31
#define MY_GL_FRAGMENT_SHADER    0x8B30
#define MY_GL_COMPILE_STATUS     0x8B81
#define MY_GL_LINK_STATUS        0x8B82
#define MY_GL_RGBA               0x1908
#define MY_GL_RGB                0x1907
#define MY_GL_FRAMEBUFFER        0x8D40
#define MY_GL_RENDERBUFFER       0x8D41
#define MY_GL_COLOR_ATTACHMENT0  0x8CE0
#define MY_GL_DEPTH_ATTACHMENT   0x8D00
#define MY_GL_TEXTURE_2D         0x0DE1
#define MY_GL_RGBA8              0x8058
#define MY_GL_DEPTH_COMPONENT24  0x81A6
#define MY_GL_FRAMEBUFFER_COMPLETE 0x8CD5
#define MY_GL_TEXTURE_MIN_FILTER 0x2801
#define MY_GL_TEXTURE_MAG_FILTER 0x2800
#define MY_GL_LINEAR             0x2601

// Function IDs (must match server)
#define GL_F_glClearColor     2015
#define GL_F_glClear          2014
#define GL_F_glViewport       2027
#define GL_F_glEnable         2022
#define GL_F_glDisable        2021
#define GL_F_glUseProgram        106
#define GL_F_glBindBuffer        109
#define GL_F_glBindVertexArray   114
#define GL_F_glEnableVertexAttribArray 112
#define GL_F_glDrawArrays        115
#define GL_F_glBindFramebuffer   119
#define GL_F_glBindTexture       122
#define GL_F_glTexParameteri     124
#define GL_F_glBindRenderbuffer  126
#define GL_F_glCompileShader     102
#define GL_F_glAttachShader      104
#define GL_F_glLinkProgram       105
#define GL_F_glDeleteShader      107
#define GL_F_glFinish            2023

#define GL_F_glCreateShader      100
#define GL_F_glCreateProgram     103
#define GL_F_glGetError          117
#define GL_F_glShaderSource      101
#define GL_F_glGenBuffers        108
#define GL_F_glGenVertexArrays   113
#define GL_F_glBufferData        110
#define GL_F_glVertexAttribPointer 111
#define GL_F_glReadPixels        116
#define GL_F_glCheckFramebufferStatus 129
#define GL_F_glGenFramebuffers   118
#define GL_F_glGenTextures       121
#define GL_F_glTexImage2D        123
#define GL_F_glGenRenderbuffers  125
#define GL_F_glRenderbufferStorage 127
#define GL_F_glFramebufferTexture2D 120
#define GL_F_glFramebufferRenderbuffer 128
#define GL_F_glGetShaderiv       130
#define GL_F_glGetShaderInfoLog  131
#define GL_F_glGetProgramiv      132
#define GL_F_glGetProgramInfoLog 133

// ── Wire helpers ────────────────────────────────────────────────────────────
static int connfd = -1;

static int send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return -1;
        p += n; len -= n;
    }
    return 0;
}

static int recv_all(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, MSG_WAITALL);
        if (n <= 0) return -1;
        p += n; len -= n;
    }
    return 0;
}

static int send_msg(int fd, uint32_t type, const void *data, uint32_t len) {
    if (send_all(fd, &type, 4) < 0) return -1;
    if (send_all(fd, &len, 4) < 0) return -1;
    if (len > 0 && send_all(fd, data, len) < 0) return -1;
    return 0;
}

static int recv_msg(int fd, uint32_t *type, std::vector<uint8_t> &payload) {
    uint32_t len;
    if (recv_all(fd, type, 4) < 0) return -1;
    if (recv_all(fd, &len, 4) < 0) return -1;
    payload.resize(len);
    if (len > 0 && recv_all(fd, payload.data(), len) < 0) return -1;
    return 0;
}

// ── Batch builder ───────────────────────────────────────────────────────────
static std::vector<uint8_t> batch_buf;
static uint32_t batch_count = 0;
static size_t batch_arg_size_offset = 0;

static void batch_begin() {
    batch_buf.clear();
    batch_buf.resize(4, 0); // reserve 4 bytes for count
    batch_count = 0;
}

static void batch_finalize_cmd() {
    if (batch_count > 0 && batch_arg_size_offset > 0) {
        uint16_t arg_size = (uint16_t)(batch_buf.size() - batch_arg_size_offset - 2);
        memcpy(batch_buf.data() + batch_arg_size_offset, &arg_size, 2);
    }
}

static void batch_cmd(uint16_t func_id) {
    batch_finalize_cmd();
    // Write func_id
    size_t off = batch_buf.size();
    batch_buf.resize(off + 4);
    memcpy(batch_buf.data() + off, &func_id, 2);
    batch_arg_size_offset = off + 2; // will be filled later
    batch_count++;
}

static void batch_write(const void *data, size_t size) {
    size_t off = batch_buf.size();
    batch_buf.resize(off + size);
    memcpy(batch_buf.data() + off, data, size);
}

template<typename T>
static void batch_arg(T val) {
    batch_write(&val, sizeof(T));
}

static int batch_flush() {
    if (batch_count == 0) return 0;
    batch_finalize_cmd();
    memcpy(batch_buf.data(), &batch_count, 4);

    if (send_msg(connfd, MSG_GL_BATCH, batch_buf.data(), batch_buf.size()) < 0)
        return -1;

    // Wait for ACK
    uint32_t ack_type;
    std::vector<uint8_t> ack_payload;
    if (recv_msg(connfd, &ack_type, ack_payload) < 0) return -1;

    batch_begin();
    return 0;
}

// ── Sync call helpers ───────────────────────────────────────────────────────
// Flush batch first, then send sync msg, receive response
static std::vector<uint8_t> sync_call(uint16_t func_id, const void *args, size_t args_size) {
    batch_flush();

    std::vector<uint8_t> payload(2 + args_size);
    memcpy(payload.data(), &func_id, 2);
    if (args_size > 0) memcpy(payload.data() + 2, args, args_size);

    send_msg(connfd, MSG_GL_SYNC, payload.data(), payload.size());

    uint32_t resp_type;
    std::vector<uint8_t> resp;
    recv_msg(connfd, &resp_type, resp);
    return resp;
}

static uint32_t sync_call_u32(uint16_t func_id, const void *args = nullptr, size_t args_size = 0) {
    auto resp = sync_call(func_id, args, args_size);
    uint32_t val = 0;
    if (resp.size() >= 4) memcpy(&val, resp.data(), 4);
    return val;
}

// ── GL command wrappers ─────────────────────────────────────────────────────
static uint32_t remote_glCreateShader(uint32_t type) {
    return sync_call_u32(GL_F_glCreateShader, &type, 4);
}

static uint32_t remote_glCreateProgram() {
    return sync_call_u32(GL_F_glCreateProgram);
}

static void remote_glShaderSource(uint32_t shader, const char *src) {
    uint32_t count = 1;
    uint32_t len = strlen(src);
    std::vector<uint8_t> args(8 + 4 + len);
    memcpy(args.data(), &shader, 4);
    memcpy(args.data()+4, &count, 4);
    memcpy(args.data()+8, &len, 4);
    memcpy(args.data()+12, src, len);
    sync_call(GL_F_glShaderSource, args.data(), args.size());
}

static uint32_t remote_glGenBuffers(int n) {
    int32_t nn = n;
    auto resp = sync_call(GL_F_glGenBuffers, &nn, 4);
    uint32_t val = 0;
    if (resp.size() >= 4) memcpy(&val, resp.data(), 4);
    return val;
}

static uint32_t remote_glGenVertexArrays(int n) {
    int32_t nn = n;
    auto resp = sync_call(GL_F_glGenVertexArrays, &nn, 4);
    uint32_t val = 0;
    if (resp.size() >= 4) memcpy(&val, resp.data(), 4);
    return val;
}

static uint32_t remote_glGenFramebuffers(int n) {
    int32_t nn = n;
    auto resp = sync_call(GL_F_glGenFramebuffers, &nn, 4);
    uint32_t val = 0;
    if (resp.size() >= 4) memcpy(&val, resp.data(), 4);
    return val;
}

static uint32_t remote_glGenTextures(int n) {
    int32_t nn = n;
    auto resp = sync_call(GL_F_glGenTextures, &nn, 4);
    uint32_t val = 0;
    if (resp.size() >= 4) memcpy(&val, resp.data(), 4);
    return val;
}

static uint32_t remote_glGenRenderbuffers(int n) {
    int32_t nn = n;
    auto resp = sync_call(GL_F_glGenRenderbuffers, &nn, 4);
    uint32_t val = 0;
    if (resp.size() >= 4) memcpy(&val, resp.data(), 4);
    return val;
}

static void remote_glBufferData(uint32_t target, const void *data, size_t data_size, uint32_t usage) {
    int64_t sz = data_size;
    std::vector<uint8_t> args(16 + data_size);
    memcpy(args.data(), &target, 4);
    memcpy(args.data()+4, &sz, 8);
    memcpy(args.data()+12, &usage, 4);
    if (data_size > 0) memcpy(args.data()+16, data, data_size);
    sync_call(GL_F_glBufferData, args.data(), args.size());
}

static void remote_glVertexAttribPointer(uint32_t index, uint32_t size, uint32_t type, uint32_t normalized, uint32_t stride, int64_t offset) {
    uint8_t args[28];
    memcpy(args, &index, 4);
    memcpy(args+4, &size, 4);
    memcpy(args+8, &type, 4);
    memcpy(args+12, &normalized, 4);
    memcpy(args+16, &stride, 4);
    memcpy(args+20, &offset, 8);
    sync_call(GL_F_glVertexAttribPointer, args, 28);
}

static void remote_glTexImage2D(uint32_t target, uint32_t level, uint32_t ifmt, uint32_t w, uint32_t h, uint32_t border, uint32_t fmt, uint32_t type, const void *pixels, size_t pixel_size) {
    std::vector<uint8_t> args(32 + pixel_size);
    memcpy(args.data(), &target, 4);
    memcpy(args.data()+4, &level, 4);
    memcpy(args.data()+8, &ifmt, 4);
    memcpy(args.data()+12, &w, 4);
    memcpy(args.data()+16, &h, 4);
    memcpy(args.data()+20, &border, 4);
    memcpy(args.data()+24, &fmt, 4);
    memcpy(args.data()+28, &type, 4);
    if (pixel_size > 0) memcpy(args.data()+32, pixels, pixel_size);
    sync_call(GL_F_glTexImage2D, args.data(), args.size());
}

static void remote_glRenderbufferStorage(uint32_t target, uint32_t ifmt, uint32_t w, uint32_t h) {
    uint8_t args[16];
    memcpy(args, &target, 4); memcpy(args+4, &ifmt, 4);
    memcpy(args+8, &w, 4); memcpy(args+12, &h, 4);
    sync_call(GL_F_glRenderbufferStorage, args, 16);
}

static void remote_glFramebufferTexture2D(uint32_t target, uint32_t attachment, uint32_t textarget, uint32_t texture, int32_t level) {
    uint8_t args[20];
    memcpy(args, &target, 4); memcpy(args+4, &attachment, 4);
    memcpy(args+8, &textarget, 4); memcpy(args+12, &texture, 4);
    memcpy(args+16, &level, 4);
    sync_call(GL_F_glFramebufferTexture2D, args, 20);
}

static void remote_glFramebufferRenderbuffer(uint32_t target, uint32_t attachment, uint32_t rbtarget, uint32_t rbo) {
    uint8_t args[16];
    memcpy(args, &target, 4); memcpy(args+4, &attachment, 4);
    memcpy(args+8, &rbtarget, 4); memcpy(args+12, &rbo, 4);
    sync_call(GL_F_glFramebufferRenderbuffer, args, 16);
}

static uint32_t remote_glCheckFramebufferStatus(uint32_t target) {
    return sync_call_u32(GL_F_glCheckFramebufferStatus, &target, 4);
}

static int32_t remote_glGetShaderiv(uint32_t shader, uint32_t pname) {
    uint8_t args[8];
    memcpy(args, &shader, 4); memcpy(args+4, &pname, 4);
    return (int32_t)sync_call_u32(GL_F_glGetShaderiv, args, 8);
}

static int32_t remote_glGetProgramiv(uint32_t program, uint32_t pname) {
    uint8_t args[8];
    memcpy(args, &program, 4); memcpy(args+4, &pname, 4);
    return (int32_t)sync_call_u32(GL_F_glGetProgramiv, args, 8);
}

static std::string remote_glGetShaderInfoLog(uint32_t shader, uint32_t max_len) {
    uint8_t args[8];
    memcpy(args, &shader, 4); memcpy(args+4, &max_len, 4);
    auto resp = sync_call(GL_F_glGetShaderInfoLog, args, 8);
    if (resp.size() < 4) return "";
    int32_t actual_len;
    memcpy(&actual_len, resp.data(), 4);
    if (actual_len <= 0 || resp.size() < (size_t)(4 + actual_len)) return "";
    return std::string((const char *)resp.data() + 4, actual_len);
}

// Read pixels from remote
struct FrameData {
    int width, height, channels;
    std::vector<uint8_t> pixels;
};

static FrameData remote_glReadPixels(int x, int y, int w, int h, uint32_t format, uint32_t type) {
    uint8_t args[24];
    int32_t ix=x, iy=y, iw=w, ih=h;
    memcpy(args, &ix, 4); memcpy(args+4, &iy, 4);
    memcpy(args+8, &iw, 4); memcpy(args+12, &ih, 4);
    memcpy(args+16, &format, 4); memcpy(args+20, &type, 4);

    // Flush batch first
    batch_flush();

    // Send sync
    std::vector<uint8_t> payload(2 + 24);
    uint16_t fid = GL_F_glReadPixels;
    memcpy(payload.data(), &fid, 2);
    memcpy(payload.data() + 2, args, 24);
    send_msg(connfd, MSG_GL_SYNC, payload.data(), payload.size());

    // Receive MSG_GL_FRAME
    uint32_t resp_type;
    std::vector<uint8_t> resp;
    recv_msg(connfd, &resp_type, resp);

    FrameData frame = {};
    if (resp_type == MSG_GL_FRAME && resp.size() >= 12) {
        memcpy(&frame.width, resp.data(), 4);
        memcpy(&frame.height, resp.data()+4, 4);
        memcpy(&frame.channels, resp.data()+8, 4);
        frame.pixels.assign(resp.begin()+12, resp.end());
    }
    return frame;
}

// ── Main: connect and render triangle ───────────────────────────────────────
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <server_ip> [port]\n", argv[0]);
        return 1;
    }

    const char *host = argv[1];
    int port = (argc > 2) ? atoi(argv[2]) : 14834;

    // Connect
    struct addrinfo hints = {}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    if (getaddrinfo(host, port_str, &hints, &res) != 0) {
        perror("getaddrinfo"); return 1;
    }
    connfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (connect(connfd, res->ai_addr, res->ai_addrlen) < 0) {
        perror("connect"); return 1;
    }
    int flag = 1;
    setsockopt(connfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int));
    freeaddrinfo(res);
    printf("Connected to %s:%d\n", host, port);

    const int W = 640, H = 480;
    batch_begin();

    // ── Setup FBO for offscreen rendering ──
    printf("Setting up FBO...\n");
    uint32_t fbo = remote_glGenFramebuffers(1);
    uint32_t color_tex = remote_glGenTextures(1);
    uint32_t depth_rbo = remote_glGenRenderbuffers(1);

    // Create color texture
    batch_cmd(GL_F_glBindTexture);
    batch_arg<uint32_t>(MY_GL_TEXTURE_2D);
    batch_arg<uint32_t>(color_tex);

    batch_cmd(GL_F_glTexParameteri);
    batch_arg<uint32_t>(MY_GL_TEXTURE_2D);
    batch_arg<uint32_t>(MY_GL_TEXTURE_MIN_FILTER);
    batch_arg<int32_t>(MY_GL_LINEAR);

    batch_cmd(GL_F_glTexParameteri);
    batch_arg<uint32_t>(MY_GL_TEXTURE_2D);
    batch_arg<uint32_t>(MY_GL_TEXTURE_MAG_FILTER);
    batch_arg<int32_t>(MY_GL_LINEAR);

    batch_flush();

    remote_glTexImage2D(MY_GL_TEXTURE_2D, 0, MY_GL_RGBA8, W, H, 0, MY_GL_RGBA, MY_GL_UNSIGNED_BYTE, nullptr, 0);

    // Create depth renderbuffer
    batch_cmd(GL_F_glBindRenderbuffer);
    batch_arg<uint32_t>(MY_GL_RENDERBUFFER);
    batch_arg<uint32_t>(depth_rbo);
    batch_flush();

    remote_glRenderbufferStorage(MY_GL_RENDERBUFFER, MY_GL_DEPTH_COMPONENT24, W, H);

    // Attach to FBO
    batch_cmd(GL_F_glBindFramebuffer);
    batch_arg<uint32_t>(MY_GL_FRAMEBUFFER);
    batch_arg<uint32_t>(fbo);
    batch_flush();

    remote_glFramebufferTexture2D(MY_GL_FRAMEBUFFER, MY_GL_COLOR_ATTACHMENT0, MY_GL_TEXTURE_2D, color_tex, 0);
    remote_glFramebufferRenderbuffer(MY_GL_FRAMEBUFFER, MY_GL_DEPTH_ATTACHMENT, MY_GL_RENDERBUFFER, depth_rbo);

    uint32_t fbo_status = remote_glCheckFramebufferStatus(MY_GL_FRAMEBUFFER);
    printf("FBO status: 0x%X (%s)\n", fbo_status,
        fbo_status == MY_GL_FRAMEBUFFER_COMPLETE ? "COMPLETE" : "ERROR");

    if (fbo_status != MY_GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "FBO not complete!\n");
        close(connfd);
        return 1;
    }

    // ── Create shaders ──
    printf("Compiling shaders...\n");

    const char *vert_src =
        "#version 330 core\n"
        "layout (location = 0) in vec2 aPos;\n"
        "layout (location = 1) in vec3 aColor;\n"
        "out vec3 vColor;\n"
        "void main() {\n"
        "    gl_Position = vec4(aPos, 0.0, 1.0);\n"
        "    vColor = aColor;\n"
        "}\n";

    const char *frag_src =
        "#version 330 core\n"
        "in vec3 vColor;\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "    FragColor = vec4(vColor, 1.0);\n"
        "}\n";

    uint32_t vs = remote_glCreateShader(MY_GL_VERTEX_SHADER);
    uint32_t fs = remote_glCreateShader(MY_GL_FRAGMENT_SHADER);
    printf("  Vertex shader: %u, Fragment shader: %u\n", vs, fs);

    remote_glShaderSource(vs, vert_src);
    batch_cmd(GL_F_glCompileShader); batch_arg<uint32_t>(vs);
    batch_flush();

    int32_t compile_ok = remote_glGetShaderiv(vs, MY_GL_COMPILE_STATUS);
    if (!compile_ok) {
        auto log = remote_glGetShaderInfoLog(vs, 1024);
        fprintf(stderr, "Vertex shader compilation failed:\n%s\n", log.c_str());
        close(connfd); return 1;
    }
    printf("  Vertex shader compiled OK\n");

    remote_glShaderSource(fs, frag_src);
    batch_cmd(GL_F_glCompileShader); batch_arg<uint32_t>(fs);
    batch_flush();

    compile_ok = remote_glGetShaderiv(fs, MY_GL_COMPILE_STATUS);
    if (!compile_ok) {
        auto log = remote_glGetShaderInfoLog(fs, 1024);
        fprintf(stderr, "Fragment shader compilation failed:\n%s\n", log.c_str());
        close(connfd); return 1;
    }
    printf("  Fragment shader compiled OK\n");

    uint32_t program = remote_glCreateProgram();
    batch_cmd(GL_F_glAttachShader); batch_arg<uint32_t>(program); batch_arg<uint32_t>(vs);
    batch_cmd(GL_F_glAttachShader); batch_arg<uint32_t>(program); batch_arg<uint32_t>(fs);
    batch_cmd(GL_F_glLinkProgram); batch_arg<uint32_t>(program);
    batch_flush();

    int32_t link_ok = remote_glGetProgramiv(program, MY_GL_LINK_STATUS);
    if (!link_ok) {
        fprintf(stderr, "Program linking failed\n");
        close(connfd); return 1;
    }
    printf("  Program linked OK (id=%u)\n", program);

    // Delete shader objects
    batch_cmd(GL_F_glDeleteShader); batch_arg<uint32_t>(vs);
    batch_cmd(GL_F_glDeleteShader); batch_arg<uint32_t>(fs);

    // ── Create triangle geometry ──
    // Positions (x,y) + Colors (r,g,b) interleaved
    float vertices[] = {
        // pos         color
         0.0f,  0.8f,  1.0f, 0.0f, 0.0f,  // top - red
        -0.8f, -0.8f,  0.0f, 1.0f, 0.0f,  // bottom left - green
         0.8f, -0.8f,  0.0f, 0.0f, 1.0f,  // bottom right - blue
    };

    uint32_t vao = remote_glGenVertexArrays(1);
    uint32_t vbo = remote_glGenBuffers(1);

    batch_cmd(GL_F_glBindVertexArray); batch_arg<uint32_t>(vao);
    batch_cmd(GL_F_glBindBuffer);
    batch_arg<uint32_t>(MY_GL_ARRAY_BUFFER);
    batch_arg<uint32_t>(vbo);
    batch_flush();

    remote_glBufferData(MY_GL_ARRAY_BUFFER, vertices, sizeof(vertices), MY_GL_STATIC_DRAW);

    // position attribute (location 0): 2 floats, stride=20, offset=0
    remote_glVertexAttribPointer(0, 2, MY_GL_FLOAT, MY_GL_FALSE, 20, 0);
    batch_cmd(GL_F_glEnableVertexAttribArray); batch_arg<uint32_t>(0);

    // color attribute (location 1): 3 floats, stride=20, offset=8
    remote_glVertexAttribPointer(1, 3, MY_GL_FLOAT, MY_GL_FALSE, 20, 8);
    batch_cmd(GL_F_glEnableVertexAttribArray); batch_arg<uint32_t>(1);

    printf("  VAO=%u, VBO=%u created\n", vao, vbo);

    // ── Render ──
    printf("Rendering triangle...\n");

    batch_cmd(GL_F_glBindFramebuffer);
    batch_arg<uint32_t>(MY_GL_FRAMEBUFFER);
    batch_arg<uint32_t>(fbo);

    batch_cmd(GL_F_glViewport);
    batch_arg<int32_t>(0); batch_arg<int32_t>(0);
    batch_arg<int32_t>(W); batch_arg<int32_t>(H);

    batch_cmd(GL_F_glClearColor);
    batch_arg<float>(0.1f); batch_arg<float>(0.1f);
    batch_arg<float>(0.15f); batch_arg<float>(1.0f);

    batch_cmd(GL_F_glClear);
    batch_arg<uint32_t>(MY_GL_COLOR_BUFFER_BIT | MY_GL_DEPTH_BUFFER_BIT);

    batch_cmd(GL_F_glUseProgram); batch_arg<uint32_t>(program);
    batch_cmd(GL_F_glBindVertexArray); batch_arg<uint32_t>(vao);
    batch_cmd(GL_F_glDrawArrays);
    batch_arg<uint32_t>(MY_GL_TRIANGLES);
    batch_arg<int32_t>(0);
    batch_arg<int32_t>(3);

    batch_cmd(GL_F_glFinish);

    // ── Read back pixels ──
    printf("Reading back pixels...\n");
    FrameData frame = remote_glReadPixels(0, 0, W, H, MY_GL_RGBA, MY_GL_UNSIGNED_BYTE);

    printf("Frame received: %dx%d, %d channels, %zu bytes\n",
        frame.width, frame.height, frame.channels, frame.pixels.size());

    // ── Verify ──
    if (frame.pixels.empty()) {
        fprintf(stderr, "FAILED: No pixel data received!\n");
        close(connfd);
        return 1;
    }

    // Check center pixel (should be part of the triangle)
    int cx = W/2, cy = H/2;
    int idx = (cy * W + cx) * frame.channels;
    uint8_t r = frame.pixels[idx], g = frame.pixels[idx+1],
            b = frame.pixels[idx+2], a = frame.pixels[idx+3];
    printf("Center pixel (%d,%d): R=%d G=%d B=%d A=%d\n", cx, cy, r, g, b, a);

    // The center of our triangle should be a mix of R+G+B (not the background 0.1,0.1,0.15)
    bool center_ok = (r > 30 || g > 30 || b > 30); // not just background

    // Check a corner (should be background color ~25,25,38)
    idx = 0;
    uint8_t br = frame.pixels[idx], bg = frame.pixels[idx+1],
            bb = frame.pixels[idx+2];
    printf("Corner pixel (0,0): R=%d G=%d B=%d\n", br, bg, bb);
    bool corner_ok = (br < 50 && bg < 50 && bb < 60);

    // Check that not all pixels are the same (rendering happened)
    bool variety = false;
    for (size_t i = 0; i + frame.channels < frame.pixels.size(); i += frame.channels * 100) {
        if (frame.pixels[i] != frame.pixels[0] ||
            frame.pixels[i+1] != frame.pixels[1]) {
            variety = true;
            break;
        }
    }

    printf("\n");
    if (center_ok && corner_ok && variety) {
        printf("========================================\n");
        printf("  SUCCESS! Triangle rendered on remote  \n");
        printf("  GPU and pixels verified locally!      \n");
        printf("========================================\n");
    } else {
        printf("FAILED: Pixel verification failed.\n");
        printf("  center_ok=%d corner_ok=%d variety=%d\n", center_ok, corner_ok, variety);
    }

    // Save raw frame as PPM for visual inspection
    FILE *f = fopen("triangle_output.ppm", "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n255\n", frame.width, frame.height);
        // PPM is RGB, we have RGBA — flip vertically (OpenGL origin is bottom-left)
        for (int row = frame.height - 1; row >= 0; row--) {
            for (int col = 0; col < frame.width; col++) {
                int i = (row * frame.width + col) * frame.channels;
                fwrite(&frame.pixels[i], 1, 3, f); // write RGB, skip A
            }
        }
        fclose(f);
        printf("Saved triangle_output.ppm\n");
    }

    close(connfd);
    return 0;
}
