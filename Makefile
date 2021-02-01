COMPUTE_V1 ?=
TUNER ?=
STEP ?= 0
CONFIG ?=
COMMIT ?=
AGENT_URL ?=
RECORD ?=
HARDWARE_CONFIG ?=
DEVICE_NAME ?=
HOST_MODE ?= 0
EXPECTED_TIMEOUT ?= inf

CPU_THREADS ?= 8
INNER_CMD = ./antares/run.sh
BACKEND = $(shell ./antares/get_backend.sh)

PARAMS ?=  docker run -v $(shell pwd):/antares -w /antares --privileged -v /:/host \
	--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
	-v $(shell dirname `ldd /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null | grep nvidia-fatbinaryloader | awk '{print $$3}'` 2>/dev/null):/usr/local/nvidia/lib64 \
	-v $(shell pwd)/public/roc_prof:/usr/local/bin/rp -e CPU_THREADS=$(CPU_THREADS) -e RECORD=$(RECORD) \
	-e STEP=$(STEP) -e AGENT_URL=$(value AGENT_URL) -e TUNER=$(TUNER) -e CONFIG='$(value CONFIG)' -e BACKEND=$(BACKEND) -e COMPUTE_V1='$(value COMPUTE_V1)' \
	-e COMMIT=$(COMMIT) -e HARDWARE_CONFIG=$(HARDWARE_CONFIG) -e DEVICE_NAME='$(value DEVICE_NAME)' -e EXPECTED_TIMEOUT=$(EXPECTED_TIMEOUT)

HTTP_PORT ?= 8880
HTTP_PREF ?= AntaresServer-$(HTTP_PORT)_
HTTP_NAME ?= $(HTTP_PREF)$(BACKEND)
HTTP_EXEC ?= $(PARAMS) -d --name=$(HTTP_NAME) -p $(HTTP_PORT):$(HTTP_PORT) antares

eval:
	@if [ "x$(HOST_MODE)" = "x0" ] && pgrep dockerd >/dev/null 2>&1; then $(MAKE) install_docker; $(PARAMS) -it --rm antares $(INNER_CMD) || true; else ./$(INNER_CMD) || true; fi

shell: install_docker
	$(PARAMS) -it --rm --network=host antares bash || true

rest-server: install_docker stop-server
	$(HTTP_EXEC) bash -c 'trap ctrl_c INT; ctrl_c() { exit 1; }; while true; do BACKEND=$(BACKEND) HTTP_SERVICE=1 HTTP_PORT=$(HTTP_PORT) $(INNER_CMD); done'

stop-server:
	$(eval CONT_NAME := $(shell docker ps | grep $(HTTP_PREF) | awk '{print $$NF}'))
	docker kill $(or $(CONT_NAME), $(CONT_NAME), $(HTTP_NAME)) >/dev/null 2>&1 || true
	docker rm $(or $(CONT_NAME), $(CONT_NAME), $(HTTP_NAME)) >/dev/null 2>&1 || true
	docker rm $(HTTP_NAME) >/dev/null 2>&1 || true

install_docker:
	docker build -t antares --network=host . -f docker/Dockerfile.$(BACKEND)*

install_host:
	./engine/install_antares_host.sh
