#!/usr/bin/env python3
"""
gl_codegen.py — OpenGL RPC client/server code generator for SCUDA.

Parses gl.xml (Khronos OpenGL Registry) and generates:
  - gen_gl_api.h       : GL function ID #defines (starting at 2000)
  - gen_gl_client.cpp  : Client-side interceptors
  - gen_gl_client.h    : Function pointer lookup table header
  - gen_gl_server.cpp  : Server-side dispatch (switch + batch handler)
  - gen_gl_server.h    : Server dispatch header

Only core GL commands (api="gl", version 1.0–4.6) are emitted.
Three categories of functions are generated:
  A. Void + all-scalar params  → batch-able (gl_batch_append / gl_batch_write)
  B. Non-void return value      → sync call (rpc_write_start_request / rpc_read)
  C. Special / pointer-heavy    → stub with // MANUAL comment
"""

from __future__ import annotations

import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GL_XML_PATH = Path(__file__).parent / "gl.xml"
OUTPUT_DIR = Path(__file__).parent

# GL function IDs start here to avoid collision with SCUDA CUDA IDs (0–1371)
GL_FUNC_ID_BASE = 2000

# Known size (bytes) of each GL scalar type
GL_TYPE_SIZES: dict[str, int] = {
    "GLenum":     4,
    "GLboolean":  1,
    "GLbitfield": 4,
    "GLbyte":     1,
    "GLubyte":    1,
    "GLshort":    2,
    "GLushort":   2,
    "GLint":      4,
    "GLuint":     4,
    "GLsizei":    4,
    "GLfloat":    4,
    "GLclampf":   4,
    "GLdouble":   8,
    "GLclampd":   8,
    "GLfixed":    4,
    "GLintptr":   8,
    "GLsizeiptr": 8,
    "GLint64":    8,
    "GLuint64":   8,
    "GLsync":     8,   # pointer-sized handle, treat as opaque 64-bit
    "GLhalf":     2,
    "GLchar":     1,
    # Less-common types that appear in core GL
    "GLclampx":   4,
    "GLint64EXT": 8,
    "GLuint64EXT":8,
    "GLhandleARB":4,
    "GLcharARB":  1,
    "GLhalfARB":  2,
    "GLintptrARB":8,
    "GLsizeiptrARB": 8,
}

# Functions that require manual, hand-written implementations.
# Codegen emits only a forward declaration with a // MANUAL comment.
MANUAL_FUNCTIONS: frozenset[str] = frozenset({
    # Texture upload / download
    "glTexImage1D", "glTexImage2D", "glTexImage3D",
    "glTexSubImage1D", "glTexSubImage2D", "glTexSubImage3D",
    "glCopyTexImage1D", "glCopyTexImage2D",
    "glGetTexImage",
    "glReadPixels",
    # Buffer data
    "glBufferData", "glBufferSubData", "glGetBufferSubData",
    "glNamedBufferData", "glNamedBufferSubData",
    "glGetNamedBufferSubData",
    # Shader source
    "glShaderSource", "glGetShaderSource",
    # Buffer mapping
    "glMapBuffer", "glMapBufferRange", "glUnmapBuffer",
    "glMapNamedBuffer", "glMapNamedBufferRange", "glUnmapNamedBuffer",
    # String queries
    "glGetString", "glGetStringi",
    # Vertex attrib pointer (offset stored in VBO, value is ptr-sized int)
    "glVertexAttribPointer", "glVertexAttribIPointer",
    "glVertexAttribLPointer",
    # Draw calls with indices pointer
    "glDrawElements", "glDrawRangeElements",
    "glDrawElementsInstanced", "glDrawElementsBaseVertex",
    "glDrawRangeElementsBaseVertex",
    "glDrawElementsInstancedBaseVertex",
    "glDrawElementsInstancedBaseInstance",
    "glDrawElementsInstancedBaseVertexBaseInstance",
    "glMultiDrawElements", "glMultiDrawElementsBaseVertex",
    "glMultiDrawElementsIndirect", "glMultiDrawElementsIndirectCount",
    # Debug callback
    "glDebugMessageCallback", "glDebugMessageCallbackARB",
    "glDebugMessageCallbackKHR",
    # Program binary
    "glGetProgramBinary", "glProgramBinary",
    # Compressed texture
    "glCompressedTexImage1D", "glCompressedTexImage2D", "glCompressedTexImage3D",
    "glCompressedTexSubImage1D", "glCompressedTexSubImage2D", "glCompressedTexSubImage3D",
    "glCompressedTextureSubImage1D", "glCompressedTextureSubImage2D",
    "glCompressedTextureSubImage3D",
    "glCompressedTextureImage1DEXT", "glCompressedTextureImage2DEXT",
    "glCompressedTextureImage3DEXT",
    # TransformFeedback varyings
    "glTransformFeedbackVaryings", "glGetTransformFeedbackVarying",
    # Uniform / attrib location helpers that take strings
    "glGetUniformLocation", "glGetAttribLocation",
    "glBindAttribLocation", "glBindFragDataLocation",
    "glBindFragDataLocationIndexed",
    "glGetFragDataLocation", "glGetFragDataIndex",
    "glGetUniformBlockIndex",
    "glGetSubroutineIndex", "glGetSubroutineUniformLocation",
    "glGetProgramResourceIndex", "glGetProgramResourceLocation",
    "glGetProgramResourceLocationIndex",
    "glGetProgramResourceName",
    # Vertex array pointer queries
    "glGetVertexAttribPointerv",
    # Sync objects (fence handle is a pointer)
    "glFenceSync",
    "glClientWaitSync", "glWaitSync", "glDeleteSync", "glIsSync",
    "glGetSynciv",
    # Generic pointer-output queries
    "glGetPointerv",
    # createShaderProgramv (takes string array)
    "glCreateShaderProgramv",
    # Shader binary
    "glShaderBinary",
    # Interop
    "glGetVkProcAddrNV",
    # Memory object / semaphore (EXT_memory_object)
    "glImportMemoryFdEXT", "glImportSemaphoreFdEXT",
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GlParam:
    name: str            # parameter name
    ptype: str           # GL type name (e.g. "GLuint"); empty for void
    is_pointer: bool     # includes any *, const *, **
    is_const: bool       # const qualifier present
    has_len: bool        # len= attribute present (variable-length array)
    raw_text: str        # full text of the param (for manual fallback)

    @property
    def is_scalar(self) -> bool:
        return not self.is_pointer

    @property
    def size(self) -> Optional[int]:
        """Return byte size for scalar types, or None if unknown/pointer."""
        if self.is_pointer:
            return None
        return GL_TYPE_SIZES.get(self.ptype)


@dataclass
class GlCommand:
    name: str
    return_type: str          # "void", "GLuint", "const GLubyte *", …
    params: list[GlParam] = field(default_factory=list)
    is_alias: bool = False    # True if this is an alias command

    @property
    def returns_void(self) -> bool:
        return self.return_type.strip() == "void"

    @property
    def returns_pointer(self) -> bool:
        return "*" in self.return_type

    @property
    def all_params_scalar(self) -> bool:
        return all(p.is_scalar for p in self.params)

    @property
    def all_scalar_sizes_known(self) -> bool:
        return all(p.size is not None for p in self.params if p.is_scalar)

    @property
    def category(self) -> str:
        """
        'special'    — in MANUAL_FUNCTIONS, has pointer params with len=, or
                       has void*/unknown-type pointer params
        'batch'      — void return, all params scalar with known sizes
        'sync'       — non-void return, all params scalar with known sizes
        'special'    — fallback for anything else
        """
        if self.name in MANUAL_FUNCTIONS:
            return "special"
        if self.returns_pointer:
            return "special"
        # Any param that is a pointer → special
        for p in self.params:
            if p.is_pointer:
                return "special"
        # All params are scalar; check that sizes are known
        if not self.all_scalar_sizes_known:
            return "special"
        if self.returns_void:
            return "batch"
        # Non-void return with scalar params
        ret_size = GL_TYPE_SIZES.get(self.return_type.strip())
        if ret_size is None:
            return "special"
        return "sync"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _element_full_text(elem: ET.Element) -> str:
    """Reconstruct the text content of an element, ignoring child tags."""
    parts = [elem.text or ""]
    for child in elem:
        parts.append(child.text or "")
        parts.append(child.tail or "")
    return "".join(parts)


def parse_param(param_elem: ET.Element) -> GlParam:
    name_elem = param_elem.find("name")
    ptype_elem = param_elem.find("ptype")

    name = name_elem.text.strip() if name_elem is not None else "unknown"
    ptype = ptype_elem.text.strip() if ptype_elem is not None else ""

    full_text = _element_full_text(param_elem).strip()
    # Normalize whitespace
    full_text = " ".join(full_text.split())

    is_pointer = "*" in full_text
    is_const = "const" in full_text
    has_len = param_elem.get("len") is not None

    return GlParam(
        name=name,
        ptype=ptype,
        is_pointer=is_pointer,
        is_const=is_const,
        has_len=has_len,
        raw_text=full_text,
    )


def parse_proto(proto_elem: ET.Element) -> tuple[str, str]:
    """Return (return_type, function_name)."""
    name_elem = proto_elem.find("name")
    fname = name_elem.text.strip() if name_elem is not None else ""

    full_text = _element_full_text(proto_elem).strip()
    full_text = " ".join(full_text.split())

    # The return type is everything before the function name
    ret = full_text
    if fname and fname in ret:
        idx = ret.rfind(fname)
        ret = ret[:idx].strip()
    return ret, fname


def parse_gl_xml(xml_path: Path) -> list[GlCommand]:
    """Parse gl.xml and return commands for core GL 1.0–4.6 only."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Step 1: collect the canonical set of core GL command names
    core_commands: set[str] = set()
    for feature in root.findall("feature"):
        if feature.get("api") != "gl":
            continue
        try:
            number = float(feature.get("number", "0"))
        except ValueError:
            continue
        if number > 4.6 + 1e-9:
            continue
        for req in feature.findall("require"):
            for cmd_ref in req.findall("command"):
                core_commands.add(cmd_ref.get("name", ""))

    # Step 2: parse all <command> elements
    commands_elem = root.find("commands")
    if commands_elem is None:
        raise RuntimeError("No <commands> section found in gl.xml")

    all_commands: dict[str, GlCommand] = {}
    for cmd_elem in commands_elem.findall("command"):
        proto = cmd_elem.find("proto")
        if proto is None:
            continue
        name_elem = proto.find("name")
        if name_elem is None:
            continue

        fname = name_elem.text.strip() if name_elem.text else ""
        if not fname:
            continue

        ret_type, _ = parse_proto(proto)
        is_alias = cmd_elem.find("alias") is not None

        params = [parse_param(p) for p in cmd_elem.findall("param")]

        all_commands[fname] = GlCommand(
            name=fname,
            return_type=ret_type,
            params=params,
            is_alias=is_alias,
        )

    # Step 3: filter to core GL only, in stable order
    result: list[GlCommand] = []
    seen: set[str] = set()

    # Iterate feature sections in version order to preserve stable ordering
    for feature in root.findall("feature"):
        if feature.get("api") != "gl":
            continue
        try:
            number = float(feature.get("number", "0"))
        except ValueError:
            continue
        if number > 4.6 + 1e-9:
            continue
        for req in feature.findall("require"):
            for cmd_ref in req.findall("command"):
                cname = cmd_ref.get("name", "")
                if cname in seen:
                    continue
                seen.add(cname)
                if cname in all_commands:
                    result.append(all_commands[cname])

    return result


# ---------------------------------------------------------------------------
# Code generation helpers
# ---------------------------------------------------------------------------

def _c_param_decl(param: GlParam) -> str:
    """Reconstruct a C parameter declaration from a GlParam."""
    return param.raw_text


def _return_default(return_type: str) -> str:
    """Return a safe default value expression for the given GL return type."""
    rt = return_type.strip()
    if rt == "void":
        return ""
    if "*" in rt:
        return "nullptr"
    defaults = {
        "GLboolean":  "GL_FALSE",
        "GLenum":     "GL_NO_ERROR",
        "GLint":      "0",
        "GLuint":     "0",
        "GLsizei":    "0",
        "GLsync":     "nullptr",
        "void *":     "nullptr",
    }
    return defaults.get(rt, "0")


# ---------------------------------------------------------------------------
# File generators
# ---------------------------------------------------------------------------

def generate_api_header(commands: list[GlCommand]) -> str:
    lines: list[str] = [
        "/* AUTO-GENERATED by gl_codegen.py — DO NOT EDIT */",
        "#ifndef GEN_GL_API_H",
        "#define GEN_GL_API_H",
        "",
        "/* GL function IDs start at 2000 to avoid collision with SCUDA CUDA IDs */",
        "",
    ]
    for idx, cmd in enumerate(commands):
        lines.append(f"#define GL_FUNC_{cmd.name} {GL_FUNC_ID_BASE + idx}")
    lines += ["", "#endif /* GEN_GL_API_H */", ""]
    return "\n".join(lines)


def generate_client_header(commands: list[GlCommand]) -> str:
    lines: list[str] = [
        "/* AUTO-GENERATED by gl_codegen.py — DO NOT EDIT */",
        "#ifndef GEN_GL_CLIENT_H",
        "#define GEN_GL_CLIENT_H",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "/**",
        " * Returns our wrapper function pointer for the given GL function name,",
        " * or nullptr if the name is not handled by this generated layer.",
        " */",
        "void *gl_get_function_pointer(const char *name);",
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        "#endif /* GEN_GL_CLIENT_H */",
        "",
    ]
    return "\n".join(lines)


def generate_server_header() -> str:
    lines: list[str] = [
        "/* AUTO-GENERATED by gl_codegen.py — DO NOT EDIT */",
        "#ifndef GEN_GL_SERVER_H",
        "#define GEN_GL_SERVER_H",
        "",
        '#include "rpc.h"',
        "",
        "/**",
        " * Dispatch a batch payload or a synchronous RPC call for the given GL",
        " * function ID.  Returns 0 on success, -1 on error.",
        " *",
        " * For batch (void/scalar) commands, payload/payload_size hold the",
        " * serialised argument bytes.  For sync commands, payload is null and",
        " * arguments are read directly from conn.",
        " */",
        "int gl_dispatch(conn_t *conn, int func_id,",
        "                const uint8_t *payload, size_t payload_size);",
        "",
        "#endif /* GEN_GL_SERVER_H */",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Client .cpp
# ---------------------------------------------------------------------------

def _gen_batch_client(cmd: GlCommand) -> str:
    """Generate a batching client wrapper for a void+scalar function."""
    param_decls = ", ".join(_c_param_decl(p) for p in cmd.params)
    lines: list[str] = [
        f"void {cmd.name}({param_decls}) {{",
        f"    gl_batch_append(GL_FUNC_{cmd.name});",
    ]
    for p in cmd.params:
        sz = p.size
        lines.append(f"    gl_batch_write(&{p.name}, sizeof({p.ptype}));")
    lines.append("}")
    return "\n".join(lines)


def _gen_sync_client(cmd: GlCommand) -> str:
    """Generate a synchronous RPC client wrapper for a function with a return value."""
    param_decls = ", ".join(_c_param_decl(p) for p in cmd.params)
    ret_type = cmd.return_type.strip()
    default_val = _return_default(ret_type)
    lines: list[str] = [
        f"{ret_type} {cmd.name}({param_decls}) {{",
        "    gl_batch_flush();",
        "    conn_t *conn = gl_get_connection();",
    ]

    # Build the rpc_write chain
    writes: list[str] = [f"rpc_write_start_request(conn, GL_FUNC_{cmd.name}) < 0"]
    for p in cmd.params:
        writes.append(f"rpc_write(conn, &{p.name}, sizeof({p.ptype})) < 0")
    writes.append("rpc_wait_for_response(conn) < 0")

    # Emit the if-chain that returns default on error
    first_line = "    if (" + writes[0]
    if len(writes) == 1:
        lines.append(first_line + f") return {default_val};")
    else:
        lines.append(first_line + " ||")
        for i, w in enumerate(writes[1:], 1):
            is_last = (i == len(writes) - 1)
            indent = "        "
            if is_last:
                lines.append(f"{indent}{w}) return {default_val};")
            else:
                lines.append(f"{indent}{w} ||")

    lines.append(f"    {ret_type} result;")
    lines.append(f"    rpc_read(conn, &result, sizeof({ret_type}));")
    lines.append("    rpc_read_end(conn);")
    lines.append("    return result;")
    lines.append("}")
    return "\n".join(lines)


def _gen_special_client(cmd: GlCommand) -> str:
    """Generate a stub declaration for a manually-implemented function."""
    param_decls = ", ".join(_c_param_decl(p) for p in cmd.params)
    ret_type = cmd.return_type.strip()
    return (
        f"// MANUAL — {cmd.name} requires hand-written implementation\n"
        f"// {ret_type} {cmd.name}({param_decls});"
    )


def generate_client_cpp(commands: list[GlCommand]) -> str:
    sections: list[str] = [
        "/* AUTO-GENERATED by gl_codegen.py — DO NOT EDIT */",
        "#include <GL/gl.h>",
        '#include "gen_gl_api.h"',
        '#include "gen_gl_client.h"',
        '#include "rpc.h"',
        "",
        "/* ---------------------------------------------------------------",
        " * These functions must be provided by the GL transport layer:",
        " *   void  gl_batch_append(int func_id);",
        " *   void  gl_batch_write(const void *data, size_t size);",
        " *   void  gl_batch_flush(void);",
        " *   conn_t *gl_get_connection(void);",
        " * --------------------------------------------------------------- */",
        "",
        "/* -----------------------------------------------------------------------",
        " *  A — Batch-able void/scalar wrappers",
        " * ----------------------------------------------------------------------- */",
    ]

    batch_cmds = [c for c in commands if c.category == "batch"]
    sync_cmds  = [c for c in commands if c.category == "sync"]
    special_cmds = [c for c in commands if c.category == "special"]

    for cmd in batch_cmds:
        sections.append("")
        sections.append(_gen_batch_client(cmd))

    sections += [
        "",
        "/* -----------------------------------------------------------------------",
        " *  B — Synchronous (return-value) wrappers",
        " * ----------------------------------------------------------------------- */",
    ]
    for cmd in sync_cmds:
        sections.append("")
        sections.append(_gen_sync_client(cmd))

    sections += [
        "",
        "/* -----------------------------------------------------------------------",
        " *  C — Special / pointer-heavy stubs  (MANUAL implementations required)",
        " * ----------------------------------------------------------------------- */",
    ]
    for cmd in special_cmds:
        sections.append("")
        sections.append(_gen_special_client(cmd))

    # Function-pointer lookup table
    sections += [
        "",
        "/* -----------------------------------------------------------------------",
        " *  Function-pointer lookup table",
        " * ----------------------------------------------------------------------- */",
        "",
        "static struct { const char *name; void *ptr; } gl_func_table[] = {",
    ]
    for cmd in commands:
        if cmd.category != "special":
            sections.append(f'    {{ "{cmd.name}", (void *){cmd.name} }},')
    sections += [
        "    { nullptr, nullptr }",
        "};",
        "",
        "void *gl_get_function_pointer(const char *name) {",
        "    for (int i = 0; gl_func_table[i].name != nullptr; ++i) {",
        "        if (__builtin_strcmp(name, gl_func_table[i].name) == 0)",
        "            return gl_func_table[i].ptr;",
        "    }",
        "    return nullptr;",
        "}",
        "",
    ]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Server .cpp
# ---------------------------------------------------------------------------

def _gen_batch_server_case(cmd: GlCommand) -> str:
    """Generate a switch case for a batch command (payload-based)."""
    lines: list[str] = [
        f"    case GL_FUNC_{cmd.name}: {{",
    ]
    offset = 0
    for p in cmd.params:
        sz = p.size
        lines.append(f"        {p.ptype} {p.name};")
        lines.append(f"        if (payload_size < {offset + sz}) return -1;")
        lines.append(f"        memcpy(&{p.name}, payload + {offset}, {sz});")
        offset += sz
    # Build the call
    args = ", ".join(p.name for p in cmd.params)
    lines.append(f"        {cmd.name}({args});")
    lines.append("        return 0;")
    lines.append("    }")
    return "\n".join(lines)


def _gen_sync_server_case(cmd: GlCommand) -> str:
    """Generate a switch case for a synchronous (return-value) command."""
    ret_type = cmd.return_type.strip()
    lines: list[str] = [f"    case GL_FUNC_{cmd.name}: {{"]
    for p in cmd.params:
        lines.append(f"        {p.ptype} {p.name};")
        lines.append(
            f"        if (rpc_read(conn, &{p.name}, sizeof({p.ptype})) < 0) return -1;"
        )
    args = ", ".join(p.name for p in cmd.params)
    lines += [
        f"        {ret_type} result = {cmd.name}({args});",
        "        if (rpc_write(conn, &result, sizeof(result)) < 0 ||",
        "            rpc_write_end(conn) < 0) return -1;",
        "        return 0;",
        "    }",
    ]
    return "\n".join(lines)


def _gen_special_server_case(cmd: GlCommand) -> str:
    """Generate a placeholder case that calls out to a manual handler."""
    return (
        f"    case GL_FUNC_{cmd.name}:\n"
        f"        /* MANUAL — see manual_gl_server.cpp */\n"
        f"        return handle_{cmd.name}_manual(conn);"
    )


def generate_server_cpp(commands: list[GlCommand]) -> str:
    batch_cmds   = [c for c in commands if c.category == "batch"]
    sync_cmds    = [c for c in commands if c.category == "sync"]
    special_cmds = [c for c in commands if c.category == "special"]

    lines: list[str] = [
        "/* AUTO-GENERATED by gl_codegen.py — DO NOT EDIT */",
        "#include <GL/gl.h>",
        "#include <string.h>",
        "#include <stdint.h>",
        '#include "gen_gl_api.h"',
        '#include "gen_gl_server.h"',
        '#include "rpc.h"',
        "",
        "/* Forward declarations for manual handlers */",
    ]
    for cmd in special_cmds:
        lines.append(f"int handle_{cmd.name}_manual(conn_t *conn);")

    lines += [
        "",
        "int gl_dispatch(conn_t *conn, int func_id,",
        "                const uint8_t *payload, size_t payload_size) {",
        "    switch (func_id) {",
        "",
        "    /* -----------------------------------------------------------",
        "     *  A — Batch commands (void/scalar, payload-based)",
        "     * ----------------------------------------------------------- */",
    ]
    for cmd in batch_cmds:
        lines.append("")
        lines.append(_gen_batch_server_case(cmd))

    lines += [
        "",
        "    /* -----------------------------------------------------------",
        "     *  B — Synchronous commands (return value)",
        "     * ----------------------------------------------------------- */",
    ]
    for cmd in sync_cmds:
        lines.append("")
        lines.append(_gen_sync_server_case(cmd))

    lines += [
        "",
        "    /* -----------------------------------------------------------",
        "     *  C — Special / pointer-heavy commands (manual)",
        "     * ----------------------------------------------------------- */",
    ]
    for cmd in special_cmds:
        lines.append("")
        lines.append(_gen_special_server_case(cmd))

    lines += [
        "",
        "    default:",
        "        return -1;",
        "    } /* switch */",
        "}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Parsing {GL_XML_PATH} …")
    commands = parse_gl_xml(GL_XML_PATH)
    print(f"  Loaded {len(commands)} core GL commands (GL 1.0–4.6)")

    batch_cmds   = [c for c in commands if c.category == "batch"]
    sync_cmds    = [c for c in commands if c.category == "sync"]
    special_cmds = [c for c in commands if c.category == "special"]

    # ---- gen_gl_api.h ----
    api_h = OUTPUT_DIR / "gen_gl_api.h"
    api_h.write_text(generate_api_header(commands), encoding="utf-8")
    print(f"  Wrote {api_h}")

    # ---- gen_gl_client.h ----
    client_h = OUTPUT_DIR / "gen_gl_client.h"
    client_h.write_text(generate_client_header(commands), encoding="utf-8")
    print(f"  Wrote {client_h}")

    # ---- gen_gl_server.h ----
    server_h = OUTPUT_DIR / "gen_gl_server.h"
    server_h.write_text(generate_server_header(), encoding="utf-8")
    print(f"  Wrote {server_h}")

    # ---- gen_gl_client.cpp ----
    client_cpp = OUTPUT_DIR / "gen_gl_client.cpp"
    client_cpp.write_text(generate_client_cpp(commands), encoding="utf-8")
    print(f"  Wrote {client_cpp}")

    # ---- gen_gl_server.cpp ----
    server_cpp = OUTPUT_DIR / "gen_gl_server.cpp"
    server_cpp.write_text(generate_server_cpp(commands), encoding="utf-8")
    print(f"  Wrote {server_cpp}")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total core GL commands : {len(commands):>5}")
    print(f"  Batch (void/scalar)    : {len(batch_cmds):>5}")
    print(f"  Sync  (return value)   : {len(sync_cmds):>5}")
    print(f"  Special / MANUAL       : {len(special_cmds):>5}")
    print()
    print("Batch functions (sample):")
    for c in batch_cmds[:10]:
        print(f"    {c.name}")
    if len(batch_cmds) > 10:
        print(f"    … and {len(batch_cmds) - 10} more")
    print()
    print("Sync functions (sample):")
    for c in sync_cmds[:10]:
        print(f"    {c.name}  -> {c.return_type}")
    if len(sync_cmds) > 10:
        print(f"    … and {len(sync_cmds) - 10} more")
    print()
    print("Special/MANUAL functions (sample):")
    for c in special_cmds[:15]:
        print(f"    {c.name}")
    if len(special_cmds) > 15:
        print(f"    … and {len(special_cmds) - 15} more")


if __name__ == "__main__":
    main()
