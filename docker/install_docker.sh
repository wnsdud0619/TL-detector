# update & upgrade
sudo apt-get update && sudo apt-get upgrade -y

# get repo
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# add docker gpg key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# add docker apt repo
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# install docker-ce
sudo apt-get update && sudo apt-get install -y docker-ce

# give sudo to user
sudo usermod -aG docker $USER

# add nvidia-docker gpg key
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# update
sudo apt-get update

# nvidia-docker install
sudo apt-get install -y nvidia-docker2 && sudo pkill -SIGHUP dockerd

echo "------------------"
echo "install finished!!"
echo "you have to reboot"
echo "------------------"
