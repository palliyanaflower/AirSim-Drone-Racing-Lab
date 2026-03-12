#!/bin/bash
# **Usage**
# - for running default image, training binaries, in windowed mode:
#     `$ ./run_docker_image.sh "" training`
# - for running default image, training binaries, in headless mode:
#     `$ ./run_docker_image.sh "" training headless`
# - for running a custom image in windowed mode, pass in your image name and tag:
#     `$ ./run_docker_image.sh DOCKER_IMAGE_NAME:TAG`
# - for running a custom image in headless mode, pass in your image name and tag, followed by "headless":
#     `$ ./run_docker_image.sh DOCKER_IMAGE_NAME:TAG headless`
# Example: ./run_docker_image.sh airsim_neurips:12.3.1-devel-ubuntu22.04

DOCKER_IMAGE_NAME=${1:-airsim_neurips:12.3.1-devel-ubuntu22.04}
IS_HEADLESS=${3:-notheadless}

# Allow all local X server connections
xhost + > /dev/null 2>&1

# Setup xauth for X11 forwarding
XAUTH=/tmp/.docker.xauth
touch $XAUTH
chmod a+r $XAUTH
if [[ -n "$DISPLAY" ]]; then
    xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge - 2>/dev/null
fi

UNREAL_BINARY_COMMAND="sudo ldconfig && bash /home/airsim_user/ADRL/ADRL/ADRL.sh -windowed -vulkan -nosound"

SDL_VIDEODRIVER_VALUE=''
if [[ $2 = "headless" ]]; then
    SDL_VIDEODRIVER_VALUE='offscreen'
fi

# pip install "numpy<2.0" --break-system-packages
docker run -it \
    --gpus all \
    --ipc=host \
    --privileged \
    --net=host \
    -v /home/kalliyanlay/Documents/BYU/research/AirSim-Drone-Racing-Lab/scripts:/home/airsim_user/AirSim-Drone-Racing-Lab/scripts \
    -v /home/kalliyanlay/Documents/BYU/research/AirSim-Drone-Racing-Lab/baselines:/home/airsim_user/AirSim-Drone-Racing-Lab/baselines \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e DISPLAY=$DISPLAY \
    -e XDG_RUNTIME_DIR=/tmp \
    -e SDL_VIDEODRIVER=$SDL_VIDEODRIVER_VALUE \
    -e SDL_HINT_CUDA_DEVICE=0 \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $XAUTH:$XAUTH:rw \
    -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:ro \
    -v /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro \
    --rm \
    $DOCKER_IMAGE_NAME \
    /bin/bash -c "$UNREAL_BINARY_COMMAND"