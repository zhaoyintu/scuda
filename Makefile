CXX = g++
CXXFLAGS = -fPIC -O2
CUDA_HEADERS ?= /tmp/cuda_headers
CUDA_HOME ?= /usr/local/cuda

# Source files
CLIENT_SRCS = client.cpp codegen/gen_client.cpp codegen/manual_client.cpp codegen/cuda_batch.cpp codegen/fatbin_cache.cpp rpc.cpp
SERVER_SRCS = server.cpp codegen/gen_server.cpp codegen/manual_server.cpp codegen/fatbin_cache.cpp rpc.cpp

# Default target
.PHONY: all local server clean deploy

all: local

# Build client library locally (no CUDA runtime needed, just headers)
local: libscuda_local.so

libscuda_local.so: $(CLIENT_SRCS) rpc.h codegen/gen_client.h
	$(CXX) -shared $(CXXFLAGS) -o $@ \
		-I$(CUDA_HEADERS) -I. \
		$(CLIENT_SRCS) \
		-ldl -lpthread -lstdc++

# Build server (requires CUDA toolkit on GPU machine)
server: scuda_server

scuda_server: $(SERVER_SRCS) rpc.h codegen/gen_server.h
	$(CXX) $(CXXFLAGS) -o $@ \
		-I. -I$(CUDA_HOME)/include \
		$(SERVER_SRCS) \
		-L$(CUDA_HOME)/lib64 \
		-ldl -lpthread -lcuda -lcudart -lnvidia-ml -lcudnn -lcublas -lstdc++

# Deploy to remote server and build
deploy:
	@if [ -z "$(REMOTE)" ]; then echo "Usage: make deploy REMOTE=user@host"; exit 1; fi
	rsync -avz --exclude='*.so' --exclude='*.o' --exclude='.git' \
		. $(REMOTE):/tmp/scuda_build/scuda/
	ssh $(REMOTE) 'cd /tmp/scuda_build/scuda && make server'

# Start server on remote
start-remote:
	@if [ -z "$(REMOTE)" ]; then echo "Usage: make start-remote REMOTE=user@host"; exit 1; fi
	ssh $(REMOTE) 'pkill -f scuda_server 2>/dev/null; sleep 1; \
		cd /tmp/scuda_build/scuda && \
		LD_LIBRARY_PATH=$(CUDA_HOME)/lib64 nohup ./scuda_server > /tmp/scuda_server.log 2>&1 & \
		sleep 1; ps aux | grep scuda_server | grep -v grep'

clean:
	rm -f libscuda_local.so scuda_server
