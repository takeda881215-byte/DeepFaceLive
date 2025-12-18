# DeepFaceLive Docker

### Requirements

* NVIDIA driver compatible with CUDA 11.8 (e.g., 520+)
* CUDA 11.8 runtime with Torch 1.13.1+cu118 and Torchvision 0.14.1+cu118

### Setup

```
git clone DeepFaceLive
cd DeepFaceLive/build/linux/
# start.sh builds and runs container with proper options, then example.sh will start DeepFaceLive with example options
./start.sh

Usage of ./start.sh
# -d specify a folder for DeepFaceLive data (videos, models, etc)
-d /home/user/DeepFaceLive_data
# -c will pass through existing video devices such as /dev/video0 and /dev/video1, etc
```
