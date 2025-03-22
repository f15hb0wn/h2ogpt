# h2oGPT

Forked from https://github.com/h2oai/h2ogpt. The intention of this fork is to be a bit more prescriptive and "turn-key" for less technical users.

## Installation & Launch
### Step 1: Configure a Docker service
* Install Docker for [Linux](https://docs.docker.com/engine/install/ubuntu/)
* Install Docker for [Windows](https://docs.docker.com/desktop/install/windows-install/)
* Install Docker for [MAC](https://docs.docker.com/desktop/install/mac-install/)

### Step 2: Enable GPU Support for Docker

Ensure docker installed and ready (requires sudo), can skip if system is already capable of running nvidia containers.  Example here is for Ubuntu, see [NVIDIA Containers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for more examples.
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit-base
sudo apt install -y nvidia-container-runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Confirm runs nvidia-smi from within docker without errors:
```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### Step 3: Mount your data library in /data_sources.
This will allow you to scan the data outside of your docker image.
You essentially want to create symbolic links to the `/data_sources` folder

### Step 4 (Optional): Mount your database directory at /db_directory
You can skip this if not running in WSL. I mount this so I can back it up off my Windows machine. 

### Step 5: Download & edit `docker-launch.sh` with any additional settings.
The run-time level changes can be made here. This script contains everything from this repository you need to launch the software.

### Step 4: Run `sudo chmod +x docker-launch.sh ; sudo ./docker-launch.sh`
This will allow launch the code. The first load will take a very long time if you have a lot of data to index.

### Step 5: After data import is complete, navigate to the URL displayed
You should be able to interact with your data at this point
