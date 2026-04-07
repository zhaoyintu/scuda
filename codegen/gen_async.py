#!/usr/bin/env python3
"""
Transform gen_client.cpp to use async RPC for queueable functions.

A function is queueable if:
1. Between rpc_wait_for_response and rpc_read_end, there's exactly ONE rpc_read
   (just for return_value - no output parameters)
2. Not in the SYNC_FUNCTIONS blacklist
3. Does not have special handling (rpc_close, etc.)

Async functions:
- Send request with RPC_ASYNC_BIT set in opcode
- Server executes but does NOT send response
- Client returns optimistic success immediately
- Errors propagate to next sync point (cudaGetLastError)
"""

import re
import sys

# Functions that MUST remain synchronous even if they have no output params
SYNC_FUNCTIONS = {
    # Initialization/shutdown
    'nvmlInit_v2', 'nvmlInitWithFlags', 'nvmlShutdown',
    'cuInit',

    # Device management - return values are checked
    'cudaSetDevice', 'cudaGetDevice', 'cudaGetDeviceCount',
    'cudaDeviceReset', 'cudaDeviceSetLimit', 'cudaDeviceGetLimit',
    'cuDeviceGet', 'cuDeviceGetCount',
    'cuCtxCreate_v2', 'cuCtxSetCurrent', 'cuCtxGetCurrent',
    'cuCtxPushCurrent_v2', 'cuCtxPopCurrent_v2', 'cuCtxDestroy_v2',

    # Error checking
    'cudaGetLastError', 'cudaPeekAtLastError',

    # Synchronization points
    'cudaDeviceSynchronize', 'cudaStreamSynchronize', 'cudaEventSynchronize',
    'cuCtxSynchronize', 'cuStreamSynchronize', 'cuEventSynchronize',

    # Stream/event query - return value indicates status
    'cudaStreamQuery', 'cudaEventQuery',
    'cuStreamQuery', 'cuEventQuery',

    # Event timing - returns elapsed time
    'cudaEventElapsedTime', 'cuEventElapsedTime',
}

# Map return type -> success value
SUCCESS_VALUES = {
    'cudaError_t': 'cudaSuccess',
    'nvmlReturn_t': 'NVML_SUCCESS',
    'CUresult': 'CUDA_SUCCESS',
    'cublasStatus_t': 'CUBLAS_STATUS_SUCCESS',
    'cudnnStatus_t': 'CUDNN_STATUS_SUCCESS',
}


def find_function_boundaries(content):
    """Find start/end positions of each function in gen_client.cpp."""
    # Match function definitions: TYPE NAME(PARAMS) {
    # The { is on the same line as the last param, params may span multiple lines
    func_pattern = re.compile(
        r'^(\w[\w\s\*]+?)\s+(\w+)\s*\([^)]*\)\s*\{',
        re.MULTILINE
    )

    matches = list(func_pattern.finditer(content))
    functions = []

    # Skip declarations like 'extern int rpc_size()' - they don't have a body
    # We detect real functions by checking if the match starts a block

    for i, match in enumerate(matches):
        func_start = match.start()
        return_type = match.group(1).strip()
        func_name = match.group(2)

        # Skip extern declarations and 'if'/'for' etc
        line_start = content.rfind('\n', 0, func_start) + 1
        line_prefix = content[line_start:func_start].strip()
        if line_prefix.startswith('extern') or line_prefix.startswith('//'):
            continue

        # Find function end: match closing brace at column 0
        # Functions end with "\n}\n" at the start of a line
        brace_depth = 0
        pos = match.end()
        brace_depth = 1  # We're inside the opening {
        while pos < len(content) and brace_depth > 0:
            if content[pos] == '{':
                brace_depth += 1
            elif content[pos] == '}':
                brace_depth -= 1
            pos += 1

        func_end = pos

        functions.append({
            'start': func_start,
            'end': func_end,
            'name': func_name,
            'return_type': return_type,
            'body': content[func_start:func_end],
        })

    return functions


def is_queueable(func):
    """Check if a function can be made async."""
    name = func['name']
    body = func['body']
    return_type = func['return_type']

    # Must be in known return type set
    if return_type not in SUCCESS_VALUES:
        return False

    # Must not be in sync blacklist
    if name in SYNC_FUNCTIONS:
        return False

    # Async for cuBLAS/cuDNN/CUDA runtime/CUDA driver API, NOT NVML
    if name.startswith('nvml'):
        return False

    # Must not have special handling
    if 'rpc_close' in body:
        return False

    # Must have the standard RPC pattern
    wait_idx = body.find('rpc_wait_for_response(conn)')
    read_end_idx = body.find('rpc_read_end(conn)')
    if wait_idx < 0 or read_end_idx < 0:
        return False

    # Count rpc_read calls between wait_for_response and rpc_read_end
    between = body[wait_idx:read_end_idx]
    rpc_reads = len(re.findall(r'rpc_read\(conn,', between))

    # Exactly 1 rpc_read = just the return_value read, no output params
    return rpc_reads == 1


def transform_function(func):
    """Transform a queueable function to use async RPC."""
    body = func['body']
    return_type = func['return_type']
    success = SUCCESS_VALUES[return_type]

    new_body = body

    # 1. Add RPC_ASYNC_BIT to the opcode
    new_body = re.sub(
        r'rpc_write_start_request\(conn, (RPC_\w+)\)',
        r'rpc_write_start_request(conn, \1 | RPC_ASYNC_BIT)',
        new_body
    )

    # 2. Replace: rpc_wait_for_response(...) || rpc_read(&return_value,...) || rpc_read_end(...)
    #    With:    rpc_write_end(conn) < 0)
    new_body = re.sub(
        r'rpc_wait_for_response\(conn\) < 0 \|\|\s*\n'
        r'\s*rpc_read\(conn, &return_value, sizeof\([^)]+\)\) < 0 \|\|\s*\n'
        r'\s*rpc_read_end\(conn\) < 0\)',
        'rpc_write_end(conn) < 0)',
        new_body
    )

    # 3. Remove return_value declaration line (2-space indent in generated code)
    #    Use negative lookahead to avoid matching "return return_value;"
    new_body = re.sub(
        r'  (?!return\b)\w+\s+return_value;\n',
        '',
        new_body
    )

    # 4. Replace 'return return_value;' with success
    new_body = new_body.replace('return return_value;', f'return {success};')

    return new_body


def main():
    input_file = 'gen_client.cpp'
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    with open(input_file, 'r') as f:
        content = f.read()

    functions = find_function_boundaries(content)

    async_funcs = []
    sync_funcs = []

    # Build replacement map
    replacements = []  # (old_body, new_body)

    for func in functions:
        if is_queueable(func):
            new_body = transform_function(func)
            replacements.append((func['body'], new_body))
            async_funcs.append(func['name'])
        else:
            sync_funcs.append(func['name'])

    # Apply replacements
    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new, 1)

    # Write output
    with open(input_file, 'w') as f:
        f.write(new_content)

    print(f"\n=== Async RPC Transformation ===")
    print(f"Total functions: {len(functions)}")
    print(f"Made async:      {len(async_funcs)}")
    print(f"Kept sync:       {len(sync_funcs)}")
    print(f"\nAsync functions ({len(async_funcs)}):")
    for name in sorted(async_funcs):
        print(f"  {name}")


if __name__ == '__main__':
    main()
