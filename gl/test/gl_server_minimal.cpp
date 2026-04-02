/**
 * Minimal GL-over-IP Server
 *
 * Creates a headless EGL context on NVIDIA GPU, listens for GL command batches,
 * executes them, and sends back rendered frames.
 *
 * Build: g++ -o gl_server gl_server_minimal.cpp -lEGL -lGL -lpthread
 * Run:   ./gl_server [port]
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <iostream>
#include <vector>

// ── Protocol constants ──────────────────────────────────────────────────────
#define MSG_GL_BATCH  1   // client→server: batched GL commands
#define MSG_GL_SYNC   2   // client→server: GL call needing return value
#define MSG_GL_FRAME  3   // server→client: rendered frame pixels
#define MSG_GL_ACK    4   // server→client: batch executed
#define MSG_EGL_INIT  10  // client→server: create EGL context
#define MSG_EGL_OK    11  // server→client: EGL context ready

// ── Wire helpers ────────────────────────────────────────────────────────────
static int send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return -1;
        p += n;
        len -= n;
    }
    return 0;
}

static int recv_all(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, MSG_WAITALL);
        if (n <= 0) return -1;
        p += n;
        len -= n;
    }
    return 0;
}

// Send a message: [type:4][length:4][payload]
static int send_msg(int fd, uint32_t type, const void *data, uint32_t len) {
    if (send_all(fd, &type, 4) < 0) return -1;
    if (send_all(fd, &len, 4) < 0) return -1;
    if (len > 0 && send_all(fd, data, len) < 0) return -1;
    return 0;
}

// Receive a message header, returns payload size. Caller must read payload.
static int recv_msg_header(int fd, uint32_t *type, uint32_t *len) {
    if (recv_all(fd, type, 4) < 0) return -1;
    if (recv_all(fd, len, 4) < 0) return -1;
    return 0;
}

// ── GL function IDs (matching gen_gl_api.h subset for triangle test) ────────
#define GL_F_glClearColor     2015
#define GL_F_glClear          2014
#define GL_F_glViewport       2027
#define GL_F_glEnable         2022
#define GL_F_glDisable        2021
#define GL_F_glDepthFunc      2024
#define GL_F_glBlendFunc      2032
#define GL_F_glClearDepth     2017
#define GL_F_glLineWidth      2003
#define GL_F_glPointSize      2004
#define GL_F_glScissor        2006
#define GL_F_glColorMask      2019
#define GL_F_glDepthMask      2020
#define GL_F_glStencilMask    2018
#define GL_F_glFinish         2023
#define GL_F_glFlush          2025

// Modern GL (loaded via eglGetProcAddress)
#define GL_F_glCreateShader      100
#define GL_F_glShaderSource      101
#define GL_F_glCompileShader     102
#define GL_F_glCreateProgram     103
#define GL_F_glAttachShader      104
#define GL_F_glLinkProgram       105
#define GL_F_glUseProgram        106
#define GL_F_glDeleteShader      107
#define GL_F_glGenBuffers        108
#define GL_F_glBindBuffer        109
#define GL_F_glBufferData        110
#define GL_F_glVertexAttribPointer 111
#define GL_F_glEnableVertexAttribArray 112
#define GL_F_glGenVertexArrays   113
#define GL_F_glBindVertexArray   114
#define GL_F_glDrawArrays        115
#define GL_F_glReadPixels        116
#define GL_F_glGetError          117
#define GL_F_glGenFramebuffers   118
#define GL_F_glBindFramebuffer   119
#define GL_F_glFramebufferTexture2D 120
#define GL_F_glGenTextures       121
#define GL_F_glBindTexture       122
#define GL_F_glTexImage2D        123
#define GL_F_glTexParameteri     124
#define GL_F_glGenRenderbuffers  125
#define GL_F_glBindRenderbuffer  126
#define GL_F_glRenderbufferStorage 127
#define GL_F_glFramebufferRenderbuffer 128
#define GL_F_glCheckFramebufferStatus 129
#define GL_F_glGetShaderiv       130
#define GL_F_glGetShaderInfoLog  131
#define GL_F_glGetProgramiv      132
#define GL_F_glGetProgramInfoLog 133
#define GL_F_glDeleteProgram     134
#define GL_F_glDeleteBuffers     135
#define GL_F_glDeleteVertexArrays 136
#define GL_F_glDeleteFramebuffers 137
#define GL_F_glDeleteRenderbuffers 138
#define GL_F_glDeleteTextures    139

// ── GL extension function pointers ──────────────────────────────────────────
typedef GLuint (*PFNGLCREATESHADERPROC)(GLenum);
typedef void (*PFNGLSHADERSOURCEPROC)(GLuint, GLsizei, const GLchar *const *, const GLint *);
typedef void (*PFNGLCOMPILESHADERPROC)(GLuint);
typedef GLuint (*PFNGLCREATEPROGRAMPROC)(void);
typedef void (*PFNGLATTACHSHADERPROC)(GLuint, GLuint);
typedef void (*PFNGLLINKPROGRAMPROC)(GLuint);
typedef void (*PFNGLUSEPROGRAMPROC)(GLuint);
typedef void (*PFNGLDELETESHADERPROC)(GLuint);
typedef void (*PFNGLGENBUFFERSPROC)(GLsizei, GLuint *);
typedef void (*PFNGLBINDBUFFERPROC)(GLenum, GLuint);
typedef void (*PFNGLBUFFERDATAPROC)(GLenum, GLsizeiptr, const void *, GLenum);
typedef void (*PFNGLVERTEXATTRIBPOINTERPROC)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void *);
typedef void (*PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint);
typedef void (*PFNGLGENVERTEXARRAYSPROC)(GLsizei, GLuint *);
typedef void (*PFNGLBINDVERTEXARRAYPROC)(GLuint);
typedef void (*PFNGLGENFRAMEBUFFERSPROC)(GLsizei, GLuint *);
typedef void (*PFNGLBINDFRAMEBUFFERPROC)(GLenum, GLuint);
typedef void (*PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum, GLenum, GLenum, GLuint, GLint);
typedef void (*PFNGLGENRENDERBUFFERSPROC)(GLsizei, GLuint *);
typedef void (*PFNGLBINDRENDERBUFFERPROC)(GLenum, GLuint);
typedef void (*PFNGLRENDERBUFFERSTORAGEPROC)(GLenum, GLenum, GLsizei, GLsizei);
typedef void (*PFNGLFRAMEBUFFERRENDERBUFFERPROC)(GLenum, GLenum, GLenum, GLuint);
typedef GLenum (*PFNGLCHECKFRAMEBUFFERSTATUSPROC)(GLenum);
typedef void (*PFNGLGETSHADERIVPROC)(GLuint, GLenum, GLint *);
typedef void (*PFNGLGETSHADERINFOLOGPROC)(GLuint, GLsizei, GLsizei *, GLchar *);
typedef void (*PFNGLGETPROGRAMIVPROC)(GLuint, GLenum, GLint *);
typedef void (*PFNGLGETPROGRAMINFOLOGPROC)(GLuint, GLsizei, GLsizei *, GLchar *);
typedef void (*PFNGLDELETEPROGRAMPROC)(GLuint);
typedef void (*PFNGLDELETEBUFFERSPROC)(GLsizei, const GLuint *);
typedef void (*PFNGLDELETEVERTEXARRAYSPROC)(GLsizei, const GLuint *);
typedef void (*PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei, const GLuint *);
typedef void (*PFNGLDELETERENDERBUFFERSPROC)(GLsizei, const GLuint *);
typedef void (*PFNGLDELETETEXTURESPROC)(GLsizei, const GLuint *);

static PFNGLCREATESHADERPROC pglCreateShader;
static PFNGLSHADERSOURCEPROC pglShaderSource;
static PFNGLCOMPILESHADERPROC pglCompileShader;
static PFNGLCREATEPROGRAMPROC pglCreateProgram;
static PFNGLATTACHSHADERPROC pglAttachShader;
static PFNGLLINKPROGRAMPROC pglLinkProgram;
static PFNGLUSEPROGRAMPROC pglUseProgram;
static PFNGLDELETESHADERPROC pglDeleteShader;
static PFNGLGENBUFFERSPROC pglGenBuffers;
static PFNGLBINDBUFFERPROC pglBindBuffer;
static PFNGLBUFFERDATAPROC pglBufferData;
static PFNGLVERTEXATTRIBPOINTERPROC pglVertexAttribPointer;
static PFNGLENABLEVERTEXATTRIBARRAYPROC pglEnableVertexAttribArray;
static PFNGLGENVERTEXARRAYSPROC pglGenVertexArrays;
static PFNGLBINDVERTEXARRAYPROC pglBindVertexArray;
static PFNGLGENFRAMEBUFFERSPROC pglGenFramebuffers;
static PFNGLBINDFRAMEBUFFERPROC pglBindFramebuffer;
static PFNGLFRAMEBUFFERTEXTURE2DPROC pglFramebufferTexture2D;
static PFNGLGENRENDERBUFFERSPROC pglGenRenderbuffers;
static PFNGLBINDRENDERBUFFERPROC pglBindRenderbuffer;
static PFNGLRENDERBUFFERSTORAGEPROC pglRenderbufferStorage;
static PFNGLFRAMEBUFFERRENDERBUFFERPROC pglFramebufferRenderbuffer;
static PFNGLCHECKFRAMEBUFFERSTATUSPROC pglCheckFramebufferStatus;
static PFNGLGETSHADERIVPROC pglGetShaderiv;
static PFNGLGETSHADERINFOLOGPROC pglGetShaderInfoLog;
static PFNGLGETPROGRAMIVPROC pglGetProgramiv;
static PFNGLGETPROGRAMINFOLOGPROC pglGetProgramInfoLog;
static PFNGLDELETEPROGRAMPROC pglDeleteProgram;
static PFNGLDELETEBUFFERSPROC pglDeleteBuffers;
static PFNGLDELETEVERTEXARRAYSPROC pglDeleteVertexArrays;
static PFNGLDELETEFRAMEBUFFERSPROC pglDeleteFramebuffers;
static PFNGLDELETERENDERBUFFERSPROC pglDeleteRenderbuffers;
static PFNGLDELETETEXTURESPROC pglDeleteTextures;

#define LOAD_GL(name) do { \
    p##name = (decltype(p##name))eglGetProcAddress(#name); \
    if (!p##name) { fprintf(stderr, "Failed to load %s\n", #name); return false; } \
} while(0)

// ── EGL setup ───────────────────────────────────────────────────────────────
static EGLDisplay egl_display = EGL_NO_DISPLAY;
static EGLContext egl_context = EGL_NO_CONTEXT;
static EGLSurface egl_surface = EGL_NO_SURFACE;

// Use EGL_EXT_platform_device for headless NVIDIA rendering
static PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT;
static PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT;

static bool init_egl(int width, int height) {
    // Load EGL extensions
    eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

    if (!eglQueryDevicesEXT || !eglGetPlatformDisplayEXT) {
        fprintf(stderr, "EGL device extensions not available\n");
        return false;
    }

    // Find NVIDIA device
    EGLDeviceEXT devices[8];
    EGLint num_devices = 0;
    eglQueryDevicesEXT(8, devices, &num_devices);
    if (num_devices == 0) {
        fprintf(stderr, "No EGL devices found\n");
        return false;
    }
    printf("Found %d EGL device(s)\n", num_devices);

    egl_display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0], NULL);
    if (egl_display == EGL_NO_DISPLAY) {
        fprintf(stderr, "eglGetPlatformDisplayEXT failed\n");
        return false;
    }

    EGLint major, minor;
    if (!eglInitialize(egl_display, &major, &minor)) {
        fprintf(stderr, "eglInitialize failed\n");
        return false;
    }
    printf("EGL %d.%d initialized\n", major, minor);

    // Choose config
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(egl_display, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
        fprintf(stderr, "eglChooseConfig failed\n");
        return false;
    }

    // Create pbuffer surface
    EGLint pbuffer_attribs[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE
    };
    egl_surface = eglCreatePbufferSurface(egl_display, config, pbuffer_attribs);

    // Bind OpenGL API
    eglBindAPI(EGL_OPENGL_API);

    // Create context (request OpenGL 3.3 core)
    EGLint context_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    egl_context = eglCreateContext(egl_display, config, EGL_NO_CONTEXT, context_attribs);
    if (egl_context == EGL_NO_CONTEXT) {
        fprintf(stderr, "eglCreateContext failed: 0x%x\n", eglGetError());
        return false;
    }

    if (!eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) {
        fprintf(stderr, "eglMakeCurrent failed\n");
        return false;
    }

    printf("OpenGL: %s\n", glGetString(GL_VERSION));
    printf("Renderer: %s\n", glGetString(GL_RENDERER));

    // Load GL extension functions
    LOAD_GL(glCreateShader);
    LOAD_GL(glShaderSource);
    LOAD_GL(glCompileShader);
    LOAD_GL(glCreateProgram);
    LOAD_GL(glAttachShader);
    LOAD_GL(glLinkProgram);
    LOAD_GL(glUseProgram);
    LOAD_GL(glDeleteShader);
    LOAD_GL(glGenBuffers);
    LOAD_GL(glBindBuffer);
    LOAD_GL(glBufferData);
    LOAD_GL(glVertexAttribPointer);
    LOAD_GL(glEnableVertexAttribArray);
    LOAD_GL(glGenVertexArrays);
    LOAD_GL(glBindVertexArray);
    LOAD_GL(glGenFramebuffers);
    LOAD_GL(glBindFramebuffer);
    LOAD_GL(glFramebufferTexture2D);
    LOAD_GL(glGenRenderbuffers);
    LOAD_GL(glBindRenderbuffer);
    LOAD_GL(glRenderbufferStorage);
    LOAD_GL(glFramebufferRenderbuffer);
    LOAD_GL(glCheckFramebufferStatus);
    LOAD_GL(glGetShaderiv);
    LOAD_GL(glGetShaderInfoLog);
    LOAD_GL(glGetProgramiv);
    LOAD_GL(glGetProgramInfoLog);
    LOAD_GL(glDeleteProgram);
    LOAD_GL(glDeleteBuffers);
    LOAD_GL(glDeleteVertexArrays);
    LOAD_GL(glDeleteFramebuffers);
    LOAD_GL(glDeleteRenderbuffers);
    LOAD_GL(glDeleteTextures);

    return true;
}

// ── Batch command dispatch ──────────────────────────────────────────────────
// Dispatches a single GL command from a batch payload.
// Returns bytes consumed, or -1 on error.
static int dispatch_batch_cmd(const uint8_t *data, size_t remaining) {
    if (remaining < 4) return -1;
    uint16_t func_id;
    uint16_t arg_size;
    memcpy(&func_id, data, 2);
    memcpy(&arg_size, data + 2, 2);

    if (remaining < (size_t)(4 + arg_size)) return -1;
    const uint8_t *args = data + 4;

    switch (func_id) {
    case GL_F_glClearColor: {
        float r, g, b, a;
        memcpy(&r, args, 4); memcpy(&g, args+4, 4);
        memcpy(&b, args+8, 4); memcpy(&a, args+12, 4);
        glClearColor(r, g, b, a);
        break;
    }
    case GL_F_glClear: {
        uint32_t mask;
        memcpy(&mask, args, 4);
        glClear(mask);
        break;
    }
    case GL_F_glViewport: {
        int32_t x, y, w, h;
        memcpy(&x, args, 4); memcpy(&y, args+4, 4);
        memcpy(&w, args+8, 4); memcpy(&h, args+12, 4);
        glViewport(x, y, w, h);
        break;
    }
    case GL_F_glEnable: {
        uint32_t cap;
        memcpy(&cap, args, 4);
        glEnable(cap);
        break;
    }
    case GL_F_glDisable: {
        uint32_t cap;
        memcpy(&cap, args, 4);
        glDisable(cap);
        break;
    }
    case GL_F_glDepthFunc: {
        uint32_t func;
        memcpy(&func, args, 4);
        glDepthFunc(func);
        break;
    }
    case GL_F_glColorMask: {
        uint8_t r = args[0], g = args[1], b = args[2], a = args[3];
        glColorMask(r, g, b, a);
        break;
    }
    case GL_F_glDepthMask: {
        uint8_t flag = args[0];
        glDepthMask(flag);
        break;
    }
    case GL_F_glUseProgram: {
        uint32_t prog;
        memcpy(&prog, args, 4);
        pglUseProgram(prog);
        break;
    }
    case GL_F_glBindBuffer: {
        uint32_t target, buffer;
        memcpy(&target, args, 4);
        memcpy(&buffer, args+4, 4);
        pglBindBuffer(target, buffer);
        break;
    }
    case GL_F_glBindVertexArray: {
        uint32_t vao;
        memcpy(&vao, args, 4);
        pglBindVertexArray(vao);
        break;
    }
    case GL_F_glEnableVertexAttribArray: {
        uint32_t index;
        memcpy(&index, args, 4);
        pglEnableVertexAttribArray(index);
        break;
    }
    case GL_F_glDrawArrays: {
        uint32_t mode;
        int32_t first, count;
        memcpy(&mode, args, 4);
        memcpy(&first, args+4, 4);
        memcpy(&count, args+8, 4);
        glDrawArrays(mode, first, count);
        break;
    }
    case GL_F_glBindFramebuffer: {
        uint32_t target, fbo;
        memcpy(&target, args, 4);
        memcpy(&fbo, args+4, 4);
        pglBindFramebuffer(target, fbo);
        break;
    }
    case GL_F_glBindTexture: {
        uint32_t target, tex;
        memcpy(&target, args, 4);
        memcpy(&tex, args+4, 4);
        glBindTexture(target, tex);
        break;
    }
    case GL_F_glBindRenderbuffer: {
        uint32_t target, rbo;
        memcpy(&target, args, 4);
        memcpy(&rbo, args+4, 4);
        pglBindRenderbuffer(target, rbo);
        break;
    }
    case GL_F_glTexParameteri: {
        uint32_t target, pname;
        int32_t param;
        memcpy(&target, args, 4);
        memcpy(&pname, args+4, 4);
        memcpy(&param, args+8, 4);
        glTexParameteri(target, pname, param);
        break;
    }
    case GL_F_glFinish: {
        glFinish();
        break;
    }
    case GL_F_glFlush: {
        glFlush();
        break;
    }
    case GL_F_glDeleteShader: {
        uint32_t shader;
        memcpy(&shader, args, 4);
        pglDeleteShader(shader);
        break;
    }
    case GL_F_glCompileShader: {
        uint32_t shader;
        memcpy(&shader, args, 4);
        pglCompileShader(shader);
        break;
    }
    case GL_F_glAttachShader: {
        uint32_t program, shader;
        memcpy(&program, args, 4);
        memcpy(&shader, args+4, 4);
        pglAttachShader(program, shader);
        break;
    }
    case GL_F_glLinkProgram: {
        uint32_t program;
        memcpy(&program, args, 4);
        pglLinkProgram(program);
        break;
    }
    default:
        fprintf(stderr, "Unknown batch func_id: %d\n", func_id);
        break;
    }

    return 4 + arg_size;
}

// ── Sync command dispatch ───────────────────────────────────────────────────
static int dispatch_sync_cmd(int connfd, const uint8_t *payload, uint32_t payload_size) {
    if (payload_size < 2) return -1;
    uint16_t func_id;
    memcpy(&func_id, payload, 2);
    const uint8_t *args = payload + 2;
    uint32_t args_size = payload_size - 2;

    switch (func_id) {
    case GL_F_glCreateShader: {
        uint32_t type;
        memcpy(&type, args, 4);
        GLuint shader = pglCreateShader(type);
        printf("  glCreateShader(%d) = %u\n", type, shader);
        uint32_t result = shader;
        send_msg(connfd, MSG_GL_ACK, &result, 4);
        break;
    }
    case GL_F_glCreateProgram: {
        GLuint program = pglCreateProgram();
        printf("  glCreateProgram() = %u\n", program);
        uint32_t result = program;
        send_msg(connfd, MSG_GL_ACK, &result, 4);
        break;
    }
    case GL_F_glGetError: {
        GLenum err = glGetError();
        uint32_t result = err;
        send_msg(connfd, MSG_GL_ACK, &result, 4);
        break;
    }
    case GL_F_glCheckFramebufferStatus: {
        uint32_t target;
        memcpy(&target, args, 4);
        GLenum status = pglCheckFramebufferStatus(target);
        uint32_t result = status;
        send_msg(connfd, MSG_GL_ACK, &result, 4);
        break;
    }
    case GL_F_glShaderSource: {
        // Custom: [shader:4][count:4][len0:4][src0...][len1:4][src1...]...
        uint32_t shader, count;
        memcpy(&shader, args, 4);
        memcpy(&count, args+4, 4);
        const uint8_t *p = args + 8;
        std::vector<std::string> sources;
        std::vector<const GLchar *> ptrs;
        for (uint32_t i = 0; i < count; i++) {
            uint32_t len;
            memcpy(&len, p, 4); p += 4;
            sources.emplace_back((const char *)p, len);
            p += len;
        }
        for (auto &s : sources) ptrs.push_back(s.c_str());
        pglShaderSource(shader, count, ptrs.data(), NULL);
        printf("  glShaderSource(shader=%u, count=%u)\n", shader, count);
        uint32_t ok = 0;
        send_msg(connfd, MSG_GL_ACK, &ok, 4);
        break;
    }
    case GL_F_glGenBuffers: {
        int32_t n;
        memcpy(&n, args, 4);
        std::vector<GLuint> bufs(n);
        pglGenBuffers(n, bufs.data());
        printf("  glGenBuffers(%d) = [%u, ...]\n", n, bufs[0]);
        send_msg(connfd, MSG_GL_ACK, bufs.data(), n * 4);
        break;
    }
    case GL_F_glGenVertexArrays: {
        int32_t n;
        memcpy(&n, args, 4);
        std::vector<GLuint> vaos(n);
        pglGenVertexArrays(n, vaos.data());
        printf("  glGenVertexArrays(%d) = [%u, ...]\n", n, vaos[0]);
        send_msg(connfd, MSG_GL_ACK, vaos.data(), n * 4);
        break;
    }
    case GL_F_glGenFramebuffers: {
        int32_t n;
        memcpy(&n, args, 4);
        std::vector<GLuint> fbos(n);
        pglGenFramebuffers(n, fbos.data());
        send_msg(connfd, MSG_GL_ACK, fbos.data(), n * 4);
        break;
    }
    case GL_F_glGenTextures: {
        int32_t n;
        memcpy(&n, args, 4);
        std::vector<GLuint> texs(n);
        glGenTextures(n, texs.data());
        send_msg(connfd, MSG_GL_ACK, texs.data(), n * 4);
        break;
    }
    case GL_F_glGenRenderbuffers: {
        int32_t n;
        memcpy(&n, args, 4);
        std::vector<GLuint> rbos(n);
        pglGenRenderbuffers(n, rbos.data());
        send_msg(connfd, MSG_GL_ACK, rbos.data(), n * 4);
        break;
    }
    case GL_F_glBufferData: {
        // [target:4][size:8][usage:4][data...]
        uint32_t target, usage;
        int64_t size;
        memcpy(&target, args, 4);
        memcpy(&size, args+4, 8);
        memcpy(&usage, args+12, 4);
        const void *data_ptr = (args_size > 16) ? (args + 16) : NULL;
        pglBufferData(target, size, data_ptr, usage);
        printf("  glBufferData(target=0x%x, size=%ld, usage=0x%x)\n", target, size, usage);
        uint32_t ok = 0;
        send_msg(connfd, MSG_GL_ACK, &ok, 4);
        break;
    }
    case GL_F_glVertexAttribPointer: {
        // [index:4][size:4][type:4][normalized:4][stride:4][offset:8]
        uint32_t index, size_param, type, normalized, stride;
        int64_t offset;
        memcpy(&index, args, 4);
        memcpy(&size_param, args+4, 4);
        memcpy(&type, args+8, 4);
        memcpy(&normalized, args+12, 4);
        memcpy(&stride, args+16, 4);
        memcpy(&offset, args+20, 8);
        pglVertexAttribPointer(index, size_param, type, normalized, stride, (const void *)offset);
        printf("  glVertexAttribPointer(index=%u, size=%u, offset=%ld)\n", index, size_param, offset);
        uint32_t ok = 0;
        send_msg(connfd, MSG_GL_ACK, &ok, 4);
        break;
    }
    case GL_F_glTexImage2D: {
        // [target:4][level:4][internalformat:4][width:4][height:4][border:4][format:4][type:4][data...]
        uint32_t target, level, ifmt, w, h, border, fmt, type;
        memcpy(&target, args, 4);
        memcpy(&level, args+4, 4);
        memcpy(&ifmt, args+8, 4);
        memcpy(&w, args+12, 4);
        memcpy(&h, args+16, 4);
        memcpy(&border, args+20, 4);
        memcpy(&fmt, args+24, 4);
        memcpy(&type, args+28, 4);
        const void *pixels = (args_size > 32) ? (args + 32) : NULL;
        glTexImage2D(target, level, ifmt, w, h, border, fmt, type, pixels);
        uint32_t ok = 0;
        send_msg(connfd, MSG_GL_ACK, &ok, 4);
        break;
    }
    case GL_F_glRenderbufferStorage: {
        uint32_t target, ifmt, w, h;
        memcpy(&target, args, 4);
        memcpy(&ifmt, args+4, 4);
        memcpy(&w, args+8, 4);
        memcpy(&h, args+12, 4);
        pglRenderbufferStorage(target, ifmt, w, h);
        uint32_t ok = 0;
        send_msg(connfd, MSG_GL_ACK, &ok, 4);
        break;
    }
    case GL_F_glFramebufferTexture2D: {
        uint32_t target, attachment, textarget, texture;
        int32_t level;
        memcpy(&target, args, 4);
        memcpy(&attachment, args+4, 4);
        memcpy(&textarget, args+8, 4);
        memcpy(&texture, args+12, 4);
        memcpy(&level, args+16, 4);
        pglFramebufferTexture2D(target, attachment, textarget, texture, level);
        uint32_t ok = 0;
        send_msg(connfd, MSG_GL_ACK, &ok, 4);
        break;
    }
    case GL_F_glFramebufferRenderbuffer: {
        uint32_t target, attachment, rbtarget, rbo;
        memcpy(&target, args, 4);
        memcpy(&attachment, args+4, 4);
        memcpy(&rbtarget, args+8, 4);
        memcpy(&rbo, args+12, 4);
        pglFramebufferRenderbuffer(target, attachment, rbtarget, rbo);
        uint32_t ok = 0;
        send_msg(connfd, MSG_GL_ACK, &ok, 4);
        break;
    }
    case GL_F_glGetShaderiv: {
        uint32_t shader, pname;
        memcpy(&shader, args, 4);
        memcpy(&pname, args+4, 4);
        GLint val;
        pglGetShaderiv(shader, pname, &val);
        int32_t result = val;
        send_msg(connfd, MSG_GL_ACK, &result, 4);
        break;
    }
    case GL_F_glGetProgramiv: {
        uint32_t program, pname;
        memcpy(&program, args, 4);
        memcpy(&pname, args+4, 4);
        GLint val;
        pglGetProgramiv(program, pname, &val);
        int32_t result = val;
        send_msg(connfd, MSG_GL_ACK, &result, 4);
        break;
    }
    case GL_F_glGetShaderInfoLog: {
        uint32_t shader, max_len;
        memcpy(&shader, args, 4);
        memcpy(&max_len, args+4, 4);
        std::vector<char> log(max_len + 1, 0);
        GLsizei actual_len = 0;
        pglGetShaderInfoLog(shader, max_len, &actual_len, log.data());
        // Send [actual_len:4][log_data...]
        std::vector<uint8_t> resp(4 + actual_len);
        memcpy(resp.data(), &actual_len, 4);
        memcpy(resp.data() + 4, log.data(), actual_len);
        send_msg(connfd, MSG_GL_ACK, resp.data(), resp.size());
        break;
    }
    case GL_F_glGetProgramInfoLog: {
        uint32_t program, max_len;
        memcpy(&program, args, 4);
        memcpy(&max_len, args+4, 4);
        std::vector<char> log(max_len + 1, 0);
        GLsizei actual_len = 0;
        pglGetProgramInfoLog(program, max_len, &actual_len, log.data());
        std::vector<uint8_t> resp(4 + actual_len);
        memcpy(resp.data(), &actual_len, 4);
        memcpy(resp.data() + 4, log.data(), actual_len);
        send_msg(connfd, MSG_GL_ACK, resp.data(), resp.size());
        break;
    }
    case GL_F_glReadPixels: {
        // [x:4][y:4][w:4][h:4][format:4][type:4]
        int32_t x, y, w, h;
        uint32_t fmt, type;
        memcpy(&x, args, 4); memcpy(&y, args+4, 4);
        memcpy(&w, args+8, 4); memcpy(&h, args+12, 4);
        memcpy(&fmt, args+16, 4); memcpy(&type, args+20, 4);
        // Calculate pixel size
        int channels = (fmt == GL_RGBA) ? 4 : (fmt == GL_RGB) ? 3 : 4;
        size_t data_size = w * h * channels;
        std::vector<uint8_t> pixels(data_size);
        glReadPixels(x, y, w, h, fmt, type, pixels.data());
        printf("  glReadPixels(%d,%d,%d,%d) -> %zu bytes\n", x, y, w, h, data_size);
        // Send as MSG_GL_FRAME: [width:4][height:4][channels:4][pixel_data...]
        std::vector<uint8_t> frame(12 + data_size);
        memcpy(frame.data(), &w, 4);
        memcpy(frame.data()+4, &h, 4);
        memcpy(frame.data()+8, &channels, 4);
        memcpy(frame.data()+12, pixels.data(), data_size);
        send_msg(connfd, MSG_GL_FRAME, frame.data(), frame.size());
        break;
    }
    default:
        fprintf(stderr, "Unknown sync func_id: %d\n", func_id);
        uint32_t err = 0xFFFFFFFF;
        send_msg(connfd, MSG_GL_ACK, &err, 4);
        break;
    }
    return 0;
}

// ── Main server loop ────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    int port = 14834;
    int width = 640, height = 480;
    if (argc > 1) port = atoi(argv[1]);

    // Initialize EGL + GL
    if (!init_egl(width, height)) {
        fprintf(stderr, "Failed to initialize EGL\n");
        return 1;
    }
    printf("EGL/GL initialized: %dx%d\n", width, height);

    // Listen for connections
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    int enable = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    listen(sockfd, 1);
    printf("GL Server listening on port %d...\n", port);

    while (true) {
        struct sockaddr_in cli;
        socklen_t cli_len = sizeof(cli);
        int connfd = accept(sockfd, (struct sockaddr *)&cli, &cli_len);
        if (connfd < 0) { perror("accept"); continue; }

        int flag = 1;
        setsockopt(connfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int));
        printf("Client connected.\n");

        // Message loop
        while (true) {
            uint32_t msg_type, msg_len;
            if (recv_msg_header(connfd, &msg_type, &msg_len) < 0) {
                printf("Client disconnected.\n");
                break;
            }

            std::vector<uint8_t> payload(msg_len);
            if (msg_len > 0 && recv_all(connfd, payload.data(), msg_len) < 0) {
                printf("Failed to read payload.\n");
                break;
            }

            if (msg_type == MSG_GL_BATCH) {
                // Parse batch: [count:4][cmd1][cmd2]...
                uint32_t count;
                memcpy(&count, payload.data(), 4);
                const uint8_t *p = payload.data() + 4;
                size_t remaining = msg_len - 4;

                for (uint32_t i = 0; i < count; i++) {
                    int consumed = dispatch_batch_cmd(p, remaining);
                    if (consumed < 0) {
                        fprintf(stderr, "Batch dispatch error at cmd %u\n", i);
                        break;
                    }
                    p += consumed;
                    remaining -= consumed;
                }
                // Send ACK
                uint32_t ok = 0;
                send_msg(connfd, MSG_GL_ACK, &ok, 4);
            } else if (msg_type == MSG_GL_SYNC) {
                dispatch_sync_cmd(connfd, payload.data(), msg_len);
            } else {
                fprintf(stderr, "Unknown message type: %u\n", msg_type);
            }
        }

        close(connfd);
    }

    close(sockfd);
    return 0;
}
