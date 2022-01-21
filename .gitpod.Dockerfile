FROM index.docker.io/library/ubuntu:latest

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends \
     sudo vim git cmake build-essential ca-certificates python3-pip liba-dev libvdpau-dev

RUN pip3 install conan
