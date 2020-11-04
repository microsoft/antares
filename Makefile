COMPUTE_V1 ?= - einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})
BACKEND ?=
TUNER ?=
STEP ?= 0
CONFIG ?=
COMMIT ?=
AGENT_URL ?=
RECORD ?=
HARDWARE_CONFIG ?=

CPU_THREADS ?= 8
INNER_CMD = ./run.sh

PARAMS ?=  docker run -v $(shell pwd):/antares -w /antares/antares --privileged -v /:/host \
	-v $(shell dirname `ldd /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null | grep nvidia-fatbinaryloader | awk '{print $$3}'` 2>/dev/null):/usr/local/nvidia/lib64 \
	-v $(shell pwd)/public/roc_prof:/usr/local/bin/rp -e CPU_THREADS=$(CPU_THREADS) -e RECORD=$(RECORD) \
	-e STEP=$(STEP) -e AGENT_URL=$(AGENT_URL) -e TUNER=$(TUNER) -e CONFIG='$(CONFIG)' -e BACKEND=$(BACKEND) -e COMPUTE_V1='$(COMPUTE_V1)' \
	-e COMMIT=$(COMMIT) -e HARDWARE_CONFIG=$(HARDWARE_CONFIG)

HTTP_PORT ?= 8880
HTTP_NAME ?= AntaresServer-$(HTTP_PORT)_$(BACKEND)
HTTP_EXEC ?= $(PARAMS) -d --name=$(HTTP_NAME) -p $(HTTP_PORT):$(HTTP_PORT) antares

eval: build
	$(PARAMS) -it --rm antares $(INNER_CMD)

shell: build
	$(PARAMS) -it --rm --network=host antares bash || true

rest-server: build stop-server
	$(HTTP_EXEC) bash -c 'trap ctrl_c INT; ctrl_c() { exit 1; }; while true; do BACKEND=$(BACKEND) HTTP_SERVICE=1 HTTP_PORT=$(HTTP_PORT) $(INNER_CMD); done'

stop-server:
	docker kill $(HTTP_NAME) >/dev/null 2>&1 || true
	docker rm $(HTTP_NAME) >/dev/null 2>&1 || true

build:
	docker build -t antares --network=host .
