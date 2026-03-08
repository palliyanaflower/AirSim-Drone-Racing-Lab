# Build using:
# docker build --network=host   --build-arg BASE_IMAGE=nvidia/cuda:12.3.1-devel-ubuntu22.04   -t airsim_neurips:12.3.1-devel-ubuntu22.04   -f Dockerfile .

# Run
# xhost +local:docker
# docker run -it --gpus all airsim_neurips:12.3.1-devel-ubuntu22.04 /bin/bash
# ./run_docker_image.sh "airsim_neurips:12.3.1-devel-ubuntu22.04" training

# Check what is missing
# docker run -it --rm \
#     --gpus all \
#     --volume="/usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro" \
#     --volume="/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:ro" \
#     airsim_neurips:12.3.1-devel-ubuntu22.04 \
#     /bin/bash -c "vulkaninfo 2>&1 | head -50"

# # Working!
# XAUTH=/tmp/.docker.xauth && \
# touch $XAUTH && \
# chmod a+r $XAUTH && \
# xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge - && \
# xhost + && \
# docker run -it \
#     --gpus all \
#     --ipc=host \
#     --privileged \
#     -e SDL_VIDEODRIVER='' \
#     -e SDL_HINT_CUDA_DEVICE='0' \
#     --net=host \
#     --env="DISPLAY=$DISPLAY" \
#     --env="XAUTHORITY=/tmp/.docker.xauth" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     --volume="/tmp/.docker.xauth:/tmp/.docker.xauth:rw" \
#     --volume="/usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro" \
#     --volume="/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:ro" \
#     --rm \
#     airsim_neurips:12.3.1-devel-ubuntu22.04 \
#     /bin/bash -c "sudo ldconfig && bash /home/airsim_user/ADRL/ADRL/ADRL.sh -windowed -vulkan"

ARG BASE_IMAGE=nvidia/cuda:12.3.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

# Consolidate into a single apt install to save layers
RUN apt-get update --fix-missing && \
	apt-get install -y --no-install-recommends \
	git \
	libglu1-mesa-dev \
	libgl1-mesa-glx \
	libglvnd-dev \
	libglvnd0 \
	libgl1 \
	libglx0 \
	libegl1 \
	libvulkan1 \
	vulkan-tools \
	mesa-vulkan-drivers \
	pulseaudio \
	python3 \
	python3-pip \
	sudo \
	wget \
	x11-xserver-utils \
	xdg-user-dirs \
	unzip \
	ca-certificates \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# # Point Vulkan to the NVIDIA ICD
# RUN mkdir -p /etc/vulkan/icd.d && \
# 	echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' \
# 	> /etc/vulkan/icd.d/nvidia_icd.json

# Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
	python3 -m pip install --no-cache-dir airsimdroneracinglab

# Create user
RUN useradd -ms /bin/bash airsim_user && \
	usermod -aG sudo,audio,video airsim_user && \
	echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER airsim_user
ENV USER=airsim_user
ENV HOME=/home/airsim_user
WORKDIR /home/airsim_user

RUN mkdir -p /home/airsim_user/Documents/AirSim

RUN git clone https://github.com/palliyanaflower/AirSim-Drone-Racing-Lab.git && \
	cd AirSim-Drone-Racing-Lab && \
	bash download_linux_binaries.sh && \
	mv ADRL/ ../ && \
	cd ../

WORKDIR /home/airsim_user