#!/bin/bash
# SCUDA - GPU-over-IP wrapper script
# Usage: ./scuda.sh [server_ip] <command>
# Example: ./scuda.sh 192.168.5.142 nvidia-smi
#          ./scuda.sh python3 -c "import torch; print(torch.cuda.is_available())"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBSCUDA="$SCRIPT_DIR/libscuda_local.so"

if [ ! -f "$LIBSCUDA" ]; then
    echo "Error: libscuda_local.so not found in $SCRIPT_DIR"
    echo "Build it first with: cd $SCRIPT_DIR && make local"
    exit 1
fi

# If SCUDA_SERVER is already set, use all args as command
if [ -n "$SCUDA_SERVER" ]; then
    exec env LD_PRELOAD="$LIBSCUDA" "$@"
fi

# Otherwise, first arg is server IP
if [ $# -lt 2 ]; then
    echo "Usage: $0 [server_ip] <command>"
    echo "   or: SCUDA_SERVER=ip $0 <command>"
    echo ""
    echo "Examples:"
    echo "  $0 192.168.5.142 nvidia-smi"
    echo "  $0 192.168.5.142 python3 -c 'import torch; print(torch.cuda.is_available())'"
    echo "  SCUDA_SERVER=192.168.5.142 $0 nvidia-smi"
    exit 1
fi

SERVER="$1"
shift
exec env SCUDA_SERVER="$SERVER" LD_PRELOAD="$LIBSCUDA" "$@"
