DOCKER_CONTAINER_NAME=v1.3
DOCKER_IMAGE_NAME=tl_classification:$DOCKER_CONTAINER_NAME
DOCKER_USER=dgist
SHARD_FOLDER_PATH=`pwd`/..
DOCKER_SHARD_FOLDER_PATH=/home/$DOCKER_USER/catkin_ws

nvidia-docker run \
       --runtime nvidia \
       --gpus all \
       -it -P --rm --name $DOCKER_CONTAINER_NAME\
       --volume="$SHARD_FOLDER_PATH:$DOCKER_SHARD_FOLDER_PATH:rw" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       -u $DOCKER_USER \
       --privileged -v /dev/bus/usb:/dev/bus/usb \
       --net=host \
       $DOCKER_IMAGE_NAME
