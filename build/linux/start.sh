#!/bin/bash

get_nv_version() {
    if [ -r /proc/driver/nvidia/version ]; then
        PROC_VER=$(grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -1)
        if [ -n "$PROC_VER" ]; then
            echo "$PROC_VER"
            return 0
        fi
    fi

    if command -v modinfo >/dev/null 2>&1; then
        MODINFO_VER=$(modinfo nvidia 2>/dev/null | awk '/^version:/ {print $2; exit}')
        if [ -n "$MODINFO_VER" ]; then
            echo "$MODINFO_VER"
            return 0
        fi
    fi

    return 1
}

NV_VERSION=$(get_nv_version) || {
    echo "Error: Unable to determine NVIDIA driver version. Ensure NVIDIA drivers are installed." >&2
    exit 1
}
NV_VER=$(echo "$NV_VERSION" | awk -F '.' '{print $1}')

DATA_FOLDER=$(pwd)/data/
declare CAM0 CAM1 CAM2 CAM3
printf "\n"
while getopts 'cd:h' opt; do
    case "$opt" in
        c)
            printf "Starting with camera devices\n"
            test -e /dev/video0 && CAM0="--device=/dev/video0:/dev/video0"
            test -e /dev/video1 && CAM1=--device=/dev/video1:/dev/video1
            test -e /dev/video2 && CAM2=--device=/dev/video2:/dev/video2
            test -e /dev/video3 && CAM3=--device=/dev/video3:/dev/video3
            echo $CAM0 $CAM1 $CAM2 $CAM3
            ;;
        d)
            DATA_FOLDER="$OPTARG"
            printf "Starting with data folder: %s\n" "$DATA_FOLDER"
            ;;
   
        ?|h)
            printf "Usage:\n$(basename $0) [-d] /path/to/your/data/folder\n"
            exit 1
            ;;
    esac
done
shift "$(($OPTIND -1))"
printf "\n"

# Warning xhost + is overly permissive and will reduce system security. Edit as desired
docker build . -t deepfacelive --build-arg NV_VER=$NV_VER
xhost +
docker run --ipc host --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $DATA_FOLDER:/data/ $CAM0 $CAM1 $CAM2 $CAM3  --rm -it deepfacelive
