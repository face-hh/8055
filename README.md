# 8055
Real-time anime vision for VR headsets.
<img alt="Watching a jschlatt video through the app" width="1565" height="1072" src="https://github.com/user-attachments/assets/ee49f016-76b6-4e33-99d0-14aa004450f8" />
<p align="center" style="color: yellowgreen;">$${\color{gray}Watching \space a \space \color{lightblue}jschlatt \space \color{gray}video  \space through  \space the \space app}$$</p>

## ⚡️ Recommended Hardware

For best results, run the project on powerful GPUs — NVIDIA H100s or similar high-end cards are ideal. Multi-GPU setups will significantly improve real-time performance. Currently, Diffusion can achieve around **10 FPS** on an H100 GPU.

**Need GPUs?**  
Get 20% off Runpod credits with code **"FACE20"** — their first discount code ever:  
https://get.runpod.io/kbudcfhdztn1

‼️ Choose a datacenter location near you. Your biggest enemy will be latency.

---

# Init
```bash
git clone https://github.com/face-hh/8055
cd 8055
```

## 1. Install Node.js and Python dependencies

```bash
# Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt install -y nodejs

# Python (Diffusion)
pip install "numpy<2.0"
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Optional: Fix blinker conflict and install remaining dependencies
apt remove python3-blinker -y
rm -rf /usr/lib/python3/dist-packages/blinker*
rm -rf /usr/local/lib/python3.10/dist-packages/blinker*
pip cache purge
pip install diffusers==0.24.0 streamdiffusion transformers pillow opencv-python safetensors "huggingface_hub>=0.30.0,<1.0"
pip install fastapi uvicorn uvloop
pip install python-multipart

# Fix diffusers import
sed -i 's/from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info/from huggingface_hub import HfFolder, hf_hub_download, model_info/' /usr/local/lib/python3.10/dist-packages/diffusers/utils/dynamic_modules_utils.py
sed -i 's/cached_download/hf_hub_download/g' /usr/local/lib/python3.10/dist-packages/diffusers/utils/dynamic_modules_utils.py

# Node.js dependencies
cd Server
npm install
```

## 2. Start the Anime Servers

### Diffusion (multi-GPU, recommended 3+ GPUs)
```bash
export TORCH_COMPILE_DISABLE=1
export PYTORCH_DISABLE_INDUCTOR=1

# Start 1 node to download model
python anime_server.py --port 8002 --gpu 0
# Once server is online, press CTRL C to exit

# Start one server per GPU (example: 8 GPUs)
nohup python anime_server.py --port 8002 --gpu 0 > anime_server_0.log 2>&1 &
nohup python anime_server.py --port 8003 --gpu 1 > anime_server_1.log 2>&1 &
nohup python anime_server.py --port 8004 --gpu 2 > anime_server_2.log 2>&1 &
nohup python anime_server.py --port 8005 --gpu 3 > anime_server_3.log 2>&1 &
nohup python anime_server.py --port 8006 --gpu 4 > anime_server_4.log 2>&1 &
nohup python anime_server.py --port 8007 --gpu 5 > anime_server_5.log 2>&1 &
nohup python anime_server.py --port 8008 --gpu 6 > anime_server_6.log 2>&1 &
nohup python anime_server.py --port 8009 --gpu 7 > anime_server_7.log 2>&1 &
```

### GAN (AnimeGANv3, single GPU only)
```bash
# Install GAN dependencies
apt update && apt install -y wget curl git ffmpeg

wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
dpkg -i cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
apt update
apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

pip install --upgrade pip
pip install onnxruntime-gpu opencv-python pillow numpy tqdm

curl -L -o deploy/AnimeGANv3_Hayao_36.onnx "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Hayao_36.onnx"

cd Server
python3 animeganv3.py

nohup python3 animeganv3.py --port 8010 > animeganv3.log 2>&1 &
```

**Note:** Only one GAN server should be running (single GPU). Diffusion can run on multiple GPUs/servers.

## 3. Run the Central Node.js Server

```bash
cd Server
node main.js
```
Your server should now be available at `http://IP:3035`. Make sure you whitelist the port in your server firewall.

Tip: to kill Diffusion nodes, run:
```bash
ps aux | grep 'python anime_server.py' | grep -v grep | awk '{print $2}' | xargs kill -9
rm anime_server_*.log
```

## 4. Sideload app in Quest
Firstly, enable **Developer Mode**. [Follow the tutorial by Meta.](https://developers.meta.com/horizon/documentation/native/android/mobile-device-setup/)

Secondly, install **Quest Link**. [Follow the tutorial by Meta.](https://www.meta.com/help/quest/509273027107091/). In your quest, connect to it via **Link** in Settings.

Lastly, sideload the app:
1. Install [Unity](https://unity.com/)
2. Import this folder
3. Install required dependencies.
4. Optional: In Unity, click "apply optimizations" if prompted.
5. Click `File > Build Settings`
    - Click "Android"
    - Select Quest in the device list
    - Click "Build and Run"

Put your headset on & it should work.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International** License (**CC BY-NC 4.0**). See the [LICENSE](LICENSE) file for details.

Made by FaceDev
