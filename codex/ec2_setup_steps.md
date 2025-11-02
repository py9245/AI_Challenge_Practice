# notebookf585393d8a EC2 셋업 가이드

새로운 Ubuntu EC2 인스턴스에서 `notebookf585393d8a.ipynb` 노트북을 학습하기 위한 단계별 설정 방법이다. 초기 점검부터 JupyterLab 접속까지 순서대로 따라 하면 된다.

## 0. 사전 점검
- 보안 그룹에 SSH(22/tcp)가 열려 있는지 확인하고, Jupyter 포트(기본 8888/tcp)는 가급적 열지 말고 SSH 터널을 사용할 것.
- 추가 EBS 볼륨을 사용한다면 장치 이름(`/dev/nvme1n1` 등)을 메모해 둔다.

## 1. 최초 접속
```bash
ssh -i <your_key>.pem ubuntu@<EC2_PUBLIC_IP>
```
- `ubuntu` 사용자 상태인지 확인하고 `sudo -l`로 권한을 점검한다.

## 2. (선택) 데이터 볼륨 마운트
```bash
sudo lsblk
sudo mkfs -t xfs /dev/nvme1n1          # 실제 장치명으로 수정
sudo mkdir -p /data
echo '/dev/nvme1n1 /data xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo mount -a
```

## 3. GPU 드라이버 설치 (GPU 인스턴스인 경우)
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms
sudo apt install -y nvidia-driver-550
sudo reboot
```
- 재부팅 후 다시 접속해 `nvidia-smi`로 드라이버를 확인한다.
- g5dn 인스턴스(A10G GPU)는 CUDA 12.x 사용 시 550 이상 드라이버가 권장된다.
- CUDA 툴킷이 필요하면 다음을 추가한다:
```bash
sudo apt install -y cuda-toolkit-12-4
```

## 4. 기본 패키지 설치
```bash
sudo apt update
sudo apt install -y git wget curl unzip zip htop tmux python3 python3-venv python3-pip
```

## 5. Miniconda 설치
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda init bash
exec bash
```

## 6. 프로젝트 가상환경 생성
- 최초 실행 시 Anaconda 채널 약관(TOS) 수락이 필요할 수 있다. 다음 명령으로 TOS를 수락한 후 진행한다:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```
```bash
conda create -n vqa python=3.11 -y
conda activate vqa
pip install --upgrade pip
```

## 7. 핵심 라이브러리 설치
- GPU(CUDA 12.4) 환경:
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```
- CPU 전용:
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```
- 노트북이 필요로 하는 주요 패키지(버전은 Kaggle 환경 참고):
```bash
pip install transformers==4.46.1 accelerate==1.1.1 einops==0.8.0 tiktoken==0.8.0 \
    huggingface_hub==0.25.2 peft==0.13.2 bitsandbytes==0.43.1 qwen-vl-utils==0.0.8 \
    datasets==3.1.0 pandas==2.2.3 scikit-learn==1.5.2
pip install jupyterlab==4.2.5 ipywidgets==8.1.3 notebook==7.2.2
```

## 8. (선택) 데이터 전송 도구
```bash
pip install gdown pydrive2
# 또는 rclone 설정
curl https://rclone.org/install.sh | sudo bash
rclone config
```

## 9. 환경 점검
```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 10. 작업 디렉터리 준비
```bash
mkdir -p ~/workspace
```
- 로컬 PC에서 노트북 업로드:
```bash
scp -i <your_key>.pem notebookf585393d8a.ipynb ubuntu@<EC2_PUBLIC_IP>:~/workspace/
```

## 11. Jupyter 기본 설정
```bash
conda activate vqa
jupyter lab --generate-config
jupyter server password        # 접속 비밀번호 설정
```

## 12. JupyterLab 실행 (tmux 권장)
```bash
tmux new -s jlab
jupyter lab --no-browser --ip 127.0.0.1 --port 8888 --NotebookApp.allow_remote_access=True
```
- `Ctrl+B`, `D`로 세션 분리.

## 13. 로컬 포트 포워딩
```bash
ssh -i <your_key>.pem -N -L 8888:localhost:8888 ubuntu@<EC2_PUBLIC_IP>
```
- 브라우저에서 `http://localhost:8888` 접속 후 비밀번호 입력.
- `~/workspace/notebookf585393d8a.ipynb` 파일을 열어 학습을 진행한다.

## 14. Hugging Face 인증 (필요 시)
```bash
huggingface-cli login
```

## 15. 추가 권장 사항
- 환경 기록:
```bash
pip freeze > ~/workspace/requirements_ec2.txt
```
- Jupyter 세션 재접속: `tmux attach -t jlab`.
- 동일한 설정을 반복해야 한다면 위 명령어를 `setup_ec2.sh` 스크립트로 정리하여 자동화한다.
