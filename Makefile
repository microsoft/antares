COMPUTE_V1 ?=
TUNER ?=
STEP ?= 0
INIT_CONFIG ?=
CONFIG ?=
COMMIT ?=
AGENT_URL ?=
RECORD ?=
HARDWARE_CONFIG ?=
DEVICE_NAME ?=
HOST_MODE ?= 0
EXPECTED_TIMEOUT ?= inf
BATCH ?=
AB_DEBUG ?= 0

CPU_THREADS ?=
INNER_CMD = ./antares/run.sh
BACKEND = $(shell ./antares/get_backend.sh)

PARAMS ?=  docker run -v $(shell pwd):/antares -w /antares --privileged -v /:/host \
	--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
	-v $(shell dirname `find /usr/lib/ -name libnvidia-ptxjitcompiler.so` 2>/dev/null | tail -n 1)/libnvidia-ptxjitcompiler.so:/usr/local/nvidia/lib64/libnvidia-ptxjitcompiler.so \
	-v $(shell dirname `find /usr/lib/ -name libcuda.so.1` 2>/dev/null | tail -n 1)/libcuda.so.1:/usr/local/nvidia/lib64/libcuda.so.1 \
	-v $(shell pwd)/.libAntares:/root/.cache/antares \
	-v $(shell pwd)/public/roc_prof:/usr/local/bin/rp -e CPU_THREADS=$(CPU_THREADS) -e ANTARES_ROOT=/antares -e BATCH=$(BATCH) -e AB_DEBUG=$(AB_DEBUG) -e INIT_CONFIG='$(value INIT_CONFIG)' \
	-e STEP=$(STEP) -e AGENT_URL='$(value AGENT_URL)' -e TUNER=$(TUNER) -e CONFIG='$(value CONFIG)' -e BACKEND=$(BACKEND) -e COMPUTE_V1='$(value COMPUTE_V1)' \
	-e COMMIT=$(COMMIT) -e HARDWARE_CONFIG=$(HARDWARE_CONFIG) -e DEVICE_NAME='$(value DEVICE_NAME)' -e EXPECTED_TIMEOUT=$(EXPECTED_TIMEOUT)

HTTP_PORT ?= 8880
HTTP_PREF ?= AntaresServer-$(HTTP_PORT)_
HTTP_NAME ?= $(HTTP_PREF)$(BACKEND)
HTTP_EXEC ?= $(PARAMS) -d --name=$(HTTP_NAME) -p $(HTTP_PORT):$(HTTP_PORT) antares

eval:
	@if [ "x$(HOST_MODE)" = "x0" ] && [ "x$(shell whoami)" = "xroot" ] && pgrep dockerd >/dev/null 2>&1 && [ -e docker/Dockerfile.$(BACKEND) ] && $(MAKE) install_docker; then $(PARAMS) -it --rm antares $(INNER_CMD) || true; else $(INNER_CMD) || true; fi

shell: install_docker
	$(PARAMS) -it --rm --network=host antares bash || true

rest-server:
	@if [ "x$(HOST_MODE)" = "x0" ] && [ "x$(shell whoami)" = "xroot" ] && pgrep dockerd >/dev/null 2>&1 && [ -e docker/Dockerfile.$(BACKEND) ] && $(MAKE) install_docker && $(MAKE) stop-server; then $(HTTP_EXEC) bash -c 'trap ctrl_c INT; ctrl_c() { exit 1; }; while true; do BACKEND=$(BACKEND) HTTP_SERVICE=1 HTTP_PORT=$(HTTP_PORT) $(INNER_CMD); done'; else HTTP_SERVICE=1 $(INNER_CMD) || true; fi

stop-server:
	$(eval CONT_NAME := $(shell docker ps | grep $(HTTP_PREF) | awk '{print $$NF}'))
	docker kill $(or $(CONT_NAME), $(CONT_NAME), $(HTTP_NAME)) >/dev/null 2>&1 || true
	docker rm $(or $(CONT_NAME), $(CONT_NAME), $(HTTP_NAME)) >/dev/null 2>&1 || true
	docker rm $(HTTP_NAME) >/dev/null 2>&1 || true

install_docker:
	docker build -t antares --network=host . -f docker/Dockerfile.$(BACKEND)

install_host:
	./engine/install_antares_host.sh

clean:
	$(INNER_CMD) clean

bdist:
	BACKEND=c-base make install_docker
	$(PARAMS) -it --rm -v $(shell pwd):/mnt antares sh -c 'cp /antares-*.whl /mnt' || true
