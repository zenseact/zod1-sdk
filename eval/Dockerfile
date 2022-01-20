FROM ubuntu

ENV TZ=Europe/Stockholm DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential libboost-all-dev cmake git

RUN cd /root &&  git clone https://github.com/zenseact/kitti_native_evaluation.git

WORKDIR /root/kitti_native_evaluation

RUN cmake ./ -DCMAKE_CXX_FLAGS="-O3" && make
