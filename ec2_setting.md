아주 좋아요 👏
이제 완전 처음부터 끝까지 — **EC2에서 딥러닝을 돌릴 수 있는 완벽한 환경 세팅 절차**를
“교사 시점으로, 원리 + 실전 명령어”로 정리해드릴게요.

지금 단계면 이미 EC2(Deep Learning용 Ubuntu 인스턴스)가 켜져 있고,
`ubuntu@ip-...` 터미널이 열려 있죠.
이제 아래 순서대로 진행하면 **PyTorch + GPU + Kaggle + JupyterLab 완성형 환경**이 됩니다 🚀

---

# 🧩 0️⃣ 기본 개념 요약

| 구분                     | 설명                            |
| ---------------------- | ----------------------------- |
| **로컬(윈도우)**            | 단지 접속/조작용 — EC2에서 실제 연산 수행    |
| **EC2 (Ubuntu 서버)**    | GPU가 달린 진짜 훈련용 머신             |
| **CUDA / NVIDIA 드라이버** | GPU와 PyTorch를 연결해주는 핵심 드라이버   |
| **Conda 가상환경**         | 프로젝트별 독립된 파이썬 실행 공간           |
| **JupyterLab**         | 브라우저에서 실험/시각화를 편하게 할 수 있는 IDE |
| **Kaggle API**         | 데이터셋 다운로드용                    |
| **W&B / Lightning**    | 실험 기록용 라이브러리 (선택)             |

---

# ✅ 1️⃣ 시스템 업데이트

```bash
sudo apt update -y && sudo apt upgrade -y
sudo apt install -y wget git vim curl unzip build-essential
```

---

# ✅ 2️⃣ NVIDIA 드라이버 설치 (GPU 필수)

> `nvidia-smi` 가 “command not found”라면 아직 설치가 안 된 상태입니다.

```bash
# 1. 설치 가능한 드라이버 확인 (Ubuntu 22.04)
apt list nvidia-driver*

# 2. PyTorch 2.4 + CUDA 12.4 기준 안정 버전
sudo apt install -y nvidia-driver-550 nvidia-utils-550
sudo reboot
```

**재부팅 후 다시 로그인하고:**

```bash
nvidia-smi
```

정상 출력 예시 👇

```
NVIDIA-SMI 550.163.01    Driver Version: 550.163.01    CUDA Version: 12.4
GPU Name: Tesla T4
```

---

# ✅ 3️⃣ Miniconda 설치 (가상환경 관리자)
---

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

---
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

# ✅ 4️⃣ Conda 가상환경 생성

```bash
conda create -n hack python=3.11 -y
conda activate hack
```

---

# ✅ 5️⃣ 필수 패키지 설치 (딥러닝 + 실험용)

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install jupyterlab kaggle timm albumentations scikit-learn pandas matplotlib seaborn lightning==2.4.0 wandb
```

---

# ✅ 6️⃣ Kaggle API 설정 (데이터 다운로드)

1️⃣ Kaggle 계정 → [Account](https://www.kaggle.com/settings/account) → “Create New API Token”
→ `kaggle.json` 다운로드
2️⃣ EC2로 업로드 (Instance Connect 이용 가능)
3️⃣ 등록:

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

4️⃣ 테스트:

```bash
kaggle datasets list | head
```

---

# ✅ 7️⃣ JupyterLab 실행 설정

```bash
jupyter lab --generate-config
vim ~/.jupyter/jupyter_lab_config.py
```

파일 안에 아래 추가:

```python
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
```

> ⚙️ **보안그룹 인바운드 규칙 추가:**
>
> * Type: Custom TCP
> * Port: 8888
> * Source: 내 IP (또는 0.0.0.0/0 테스트용)

실행:

```bash
nohup jupyter lab --port=8888 --no-browser --allow-root &
```

터미널에 나오는 `token=...` 부분 복사해서
👉 브라우저에서
`http://<EC2 퍼블릭IP>:8888/lab?token=...`
로 접속.

---

# ✅ 8️⃣ GPU 연결 테스트

```bash
python - <<'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
EOF
```

정상이라면 👇

```
Torch version: 2.4.0
CUDA available: True
Device: Tesla T4
CUDA version: 12.4
```

---

# ✅ 9️⃣ 프로젝트 폴더 구성

```bash
mkdir -p ~/proj/{data,src,outputs,ckpt,notebooks}
cd ~/proj
```

---

# ✅ 10️⃣ 간단한 딥러닝 테스트

```python
# jupyter에서 실행
import torch, torchvision
model = torchvision.models.resnet18()
x = torch.randn(1,3,224,224).cuda()
print(model(x).shape)
```

→ GPU에서 1배치 연산 성공하면 OK.

---

# ✅ 11️⃣ (선택) 실험 로그 관리

```bash
wandb login
```

* [Weights & Biases](https://wandb.ai/) 에서 API Key 복사
* 실험 기록 / 그래프 시각화 자동화 가능

---

# 🧠 정리 — “딥러닝 돌릴 수 있는 완성 환경”

| 항목         | 도구                    | 상태            |
| ---------- | --------------------- | ------------- |
| GPU 드라이버   | NVIDIA 550            | ✅ 설치          |
| CUDA       | 12.4                  | ✅ PyTorch 호환  |
| PyTorch    | 2.4.0 (cu124)         | ✅ GPU 인식      |
| JupyterLab | 8888 포트               | ✅ 브라우저 접속     |
| Kaggle API | ~/.kaggle/kaggle.json | ✅ 데이터 다운로드 가능 |
| 실험 관리      | wandb                 | ⚙️ 선택적        |

---

📌 **결론 요약 한 줄**

> → `nvidia-smi` / `torch.cuda.is_available()` 둘 다 OK면 딥러닝 바로 돌릴 수 있습니다 💪

---

원하신다면
🔥 이 모든 단계를 자동으로 세팅해주는
`setup_dl_env.sh` (복사 → 붙여넣기 한 번으로 끝) 버전으로 만들어드릴까요?
그럼 EC2 초기화할 때마다 한 줄로 딥러닝 환경이 완성됩니다.
