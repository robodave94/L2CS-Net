FROM rrdavidh/humble_cuda_dev_image:cv_runtime

# Remove old ROS key and source list
RUN rm -f /etc/apt/trusted.gpg.d/ros-archive-keyring.gpg || true \
    && apt-key del F42ED6FBAB17C654 || true \
    && rm -f /etc/apt/sources.list.d/ros2.list /etc/apt/sources.list.d/ros-latest.list || true

# Install latest ros-apt-source for ROS 2
RUN apt-get update && apt-get install -y curl \
    && export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') \
    && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" \
    && apt-get install -y /tmp/ros2-apt-source.deb

# Update the package list
RUN apt clean && apt update -y

 #Setup custom args for compilation and operation
ARG dev_user=vscode
ARG ros_ws_name=gaze_ws
ARG ROS_DISTRO=humble
ARG RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ARG ROS_DOMAIN_ID=54

ENV RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}
ENV ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
ENV ROS_DISTRO=${ROS_DISTRO}

## set the root passwork to root for dev/debugging
RUN echo "root:root" | chpasswd

## Add extra dependancies for the dev container
RUN apt-get install -y \
    ros-${ROS_DISTRO}-camera-info-manager \
    usbutils \
    uuid-dev

#add v4l2 drivers
RUN apt update -y
RUN apt install v4l-utils -y
RUN apt install ros-humble-v4l2-camera -y  

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y build-essential cmake git pkg-config
RUN apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
RUN apt-get install -y libatlas-base-dev 
#libjasper-dev
RUN apt-get install -y --no-install-recommends libboost-all-dev
RUN apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
# RUN apt-get install -y python-pip
# RUN apt-get install -y python-dev
# RUN apt-get install -y python-numpy python-scipy
RUN apt-get install -y libopencv-dev unzip wget

RUN pip install --upgrade pip
RUN pip install numpy scipy scikit-learn matplotlib deepsort motpy

RUN useradd -ms /bin/bash ${dev_user}
RUN usermod -u 1000 ${dev_user} 
RUN usermod -g 1000 ${dev_user}

## Add the sourcing to the bashrc file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${dev_user}/.bashrc
RUN echo "source /home/${dev_user}/${ros_ws_name}/install/setup.bash" >> /home/${dev_user}/.bashrc
# Add the sourcing to the users .profile file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${dev_user}/.profile
RUN echo "source /home/${dev_user}/${ros_ws_name}/install/setup.bash" >> /home/${dev_user}/.profile

### Set the user ${dev_user} to the owner of their own directory
RUN mkdir -p /home/${dev_user}/${ros_ws_name} 
RUN chown -R ${dev_user}:${dev_user} /home/${dev_user}/${ros_ws_name} 
RUN chmod -R u+rwx /home/${dev_user}/${ros_ws_name} 

RUN mkdir -p /home/tools
WORKDIR /home/tools
RUN git clone https://github.com/robodave94/L2CS-Net.git
WORKDIR /home/tools/L2CS-Net
RUN pip install .