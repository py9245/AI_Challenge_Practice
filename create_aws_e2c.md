
# SSAFY 해커톤 초보자를 위한 **AWS EC2 100달러 세팅 가이드**

> 목적: **8일 이내 / 예산 100달러로** 딥러닝 학습 환경을 안정적으로 구축하고, 중단/재개가 가능한 워크플로를 만든다.  
> 대상: AWS **처음** 쓰는 사람. 클릭 위치까지 따라하면 끝.

---

## 0) 요약 체크리스트 (바로 쓰는 사람용)

- [ ] **리전**: `us-east-1 (N. Virginia)` 선택 → **가장 저렴, 재고 많음**
- [ ] **한도(Quota) 상승**: EC2 **G and VT vCPU** 를 `8`로 요청 (승인 후 진행)
- [ ] **인스턴스**: `g4dn.xlarge (T4 16GB)` **Spot** 사용
- [ ] **스토리지(EBS)**: 100~200GB (데이터/체크포인트 보관)
- [ ] **AMI**: Deep Learning AMI (Ubuntu 22.04)
- [ ] **보안그룹**: SSH(22)만 허용, Jupyter(8888)는 필요 시 내 IP만 허용
- [ ] **비용통제**: Spot + 매일 **중지(Stop)**, EBS만 유지. `Billing Alarm` 설정.
- [ ] **학습재개**: 체크포인트를 `/home/ubuntu/checkpoints` 또는 S3에 주기 저장.

---

## 1) 왜 `us-east-1` 인가?

- **요금이 가장 저렴**하고 **GPU 가용성**이 높다.  
- 서울(`ap-northeast-2`) 대비 동일 인스턴스가 **30~40%** 저렴한 경우가 많다.  
- 네트워크 지연은 학습 성능에 큰 영향 없음(연산은 GPU 내에서 수행).

> 콘솔 우상단 리전 선택: **N. Virginia (us-east-1)**

---

## 2) vCPU 한도(Quota) 올리기

1. 상단 검색 → **Service Quotas** → **EC2** 선택  
2. 검색창에 `vCPU` 입력 후 아래 항목 확인  
   - **Running On-Demand G and VT instances** (GPU g4dn/g5 등)
3. 각 항목에서 **Request quota increase** 클릭  
   - **New limit value**: `8` 추천 (g4dn.xlarge = 4 vCPU → 2대 가능)
   - 설명(영문 예시):
     ```text
     I am a student participating in an AI hackathon (SSAFY). 
     I need to launch a g4dn.xlarge instance for deep learning training.
     This is for educational and research purposes.
     ```
4. 승인 이메일 수신 후 다음 단계로 진행.

> **Tip**: 한도는 **리전별**로 관리된다. 반드시 `us-east-1`에서 요청.

---

## 3) 예산 설계 (100달러 목표)

| 선택 | 시간당 | 8일 24시간 | 전략 |
| --- | ---: | ---: | --- |
| **g4dn.xlarge (On-Demand)** | \$0.526 | \$101 | 예산 근접, 권장 X |
| **g4dn.xlarge (Spot)** | ~\$0.14–0.20 | \$27–\$38 | ✅ **권장** |
| **g5.xlarge (A10G)** | ~\$1.00 | \$192 | 예산 초과 |

**권장 시나리오:** `g4dn.xlarge (Spot)` + 매일 10~12시간만 가동 + EBS 유지 → **\$20~40** 수준.

---

## 4) EC2 인스턴스 생성 (Spot)

1. EC2 콘솔 → **Instances** → **Launch instances**
2. **Name**: `ssafy_ai_challenge`
3. **Application and OS Images**: **Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)** 선택  
   - (이름이 길어도 DLAMI + Ubuntu 22.04 / PyTorch 포함이면 OK)
4. **Instance type**: `g4dn.xlarge`
5. **Key pair (login)**: 새로 생성 (예: `ssafy-key`) → `.pem` 안전 보관
6. **Network settings**:  
   - **Security group**: 새로 생성  
   - **Inbound rules**:  
     - `SSH` 포트 `22` **My IP** 로 제한  
     - (선택) `Custom TCP` 포트 `8888` **My IP** (Jupyter 노트북 용)
7. **Configure storage**:  
   - 루트 볼륨 `100~200GB`, 타입 `gp3`
8. **Advanced details** 펼치기 → `Purchasing option`에서 **Request Spot instances** 체크  
   - Interruption behavior: `Stop` 또는 `Terminate` → **Stop 권장**(재개 수월)  
   - (선택) 최대 가격 지정(기본 자동도 OK)
9. **Launch instance**

> **중요**: Spot은 회수(Interruption) 가능. **체크포인트를 자주 저장**하는 워크플로가 필수.

---

## 5) 접속 & 기본 확인

### 5.1 SSH 접속
```bash
chmod 400 ssafy-key.pem
ssh -i ssafy-key.pem ubuntu@<EC2_PUBLIC_DNS>
```

### 5.2 GPU 확인
```bash
nvidia-smi
```

### 5.3 Conda & PyTorch 확인 (DLAMI 기본 포함)
```bash
conda env list
python - << 'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

---

## 6) Jupyter 노트북(선택)

### 6.1 패스워드/토큰 방식 설정
```bash
# 가상환경 예시 (base 사용 가능)
pip install jupyter --upgrade --user

# Jupyter 실행
jupyter notebook --no-browser --port 8888
```

### 6.2 로컬에서 포트 포워딩
```bash
ssh -i ssafy-key.pem -L 8888:localhost:8888 ubuntu@<EC2_PUBLIC_DNS>
```
브라우저에서 `http://localhost:8888` 접속 → 토큰 입력.

> **보안 주의**: 보안그룹에서 8888을 0.0.0.0/0으로 열지 말 것. 필요 시 **My IP**로만 허용.

---

## 7) 데이터 & 체크포인트 관리

### 7.1 디렉토리 구조
```bash
mkdir -p ~/data ~/checkpoints ~/logs
```

### 7.2 S3 사용 (선택)
```bash
# AWS CLI 설정
aws configure  # Access Key / Secret / Region(us-east-1)

# 업로드/다운로드
aws s3 sync ~/checkpoints s3://<your-bucket>/checkpoints --exact-timestamps
aws s3 sync s3://<your-bucket>/data ~/data --no-progress
```

### 7.3 학습 코드에서 체크포인트 저장 (PyTorch 예시)
```python
# 매 N step 마다
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "step": step,
}, f"/home/ubuntu/checkpoints/ckpt_{epoch}_{step}.pt")
```

> **Spot 회수/Stop 대비**: 저장 주기를 짧게, 파일명에 시간/스텝 포함.

---

## 8) 비용 절약 루틴

- ✅ **학습 끝나면 무조건 `Stop`** (중지). **Terminate**(종료) 하면 EBS까지 삭제될 수 있으니 주의.  
- ✅ **EBS는 유지**: 다음날 **Start**로 같은 디스크로 재개 가능.  
- ✅ **CloudWatch Billing Alarm** 만들기 (초과 방지).  
- ✅ 큰 업로드/다운로드는 학습 **전/후 한 번**에 처리.  
- ✅ 필요 시 `tmux`/`screen`으로 세션 유지:
  ```bash
  sudo apt-get update && sudo apt-get install -y tmux
  tmux new -s train
  # (학습 시작)
  # 분리: Ctrl+b, d / 재접속: tmux attach -t train
  ```

---

## 9) Billing Alarm 설정 (필수)

1. 콘솔 → **CloudWatch** → **Billing** → **Create alarm**
2. Metric: **Total Estimated Charges (USD)**
3. 조건: **>= 80** (또는 원하는 금액)
4. 알림: 이메일 구독 설정

---

## 10) 보안 필수 체크

- **MFA** 활성화(루트 계정)  
- **IAM 사용자** 생성 후 사용 (루트로 작업하지 않기)  
- **키(`.pem`)** 외부 유출 금지  
- **보안그룹**은 항상 **최소 허용(IP 제한)**

---

## 11) Spot 회수 대응 팁

- **Interruption notice**는 회수 2분 전 전달(메타데이터).  
- 실무에서는 데몬/콜백으로 신호를 받아 **즉시 체크포인트 저장**.  
  간단 버전: 짧은 `save_every_steps`로 대응.

---

## 12) 자주 하는 실수 Top 7

1. 리전이 서울이라 비용 과다 → **us-east-1**로 전환
2. vCPU Quota 미승인 상태에서 생성 시도 → **먼저 승인**
3. 인스턴스 **Terminate**로 날려버림 → **Stop**만 눌러라
4. 보안그룹 0.0.0.0/0 전체 오픈 → **My IP**로 제한
5. 체크포인트 미저장 → Spot 회수 시 **학습 날림**
6. EBS 30GB 기본값 → 데이터 부족/I/O 병목 → **100GB+**
7. Billing Alarm 미설정 → **예산 초과**

---

## 13) 종료/재개 워크플로 (매일 반복)

1) **학습 전**: `Start` → SSH 접속 → `nvidia-smi` 확인  
2) **학습**: tmux에서 실행, `~/checkpoints` 주기 저장  
3) **학습 후**: 로그/체크포인트 확인 → 필요 시 `aws s3 sync` → **Stop**  
4) **다음날**: `Start` → 같은 볼륨/환경으로 재개

---

## 14) 최소 명령어 치트시트

```bash
# 접속
ssh -i ssafy-key.pem ubuntu@ec2-xx-xx-xx-xx.compute-1.amazonaws.com

# GPU
nvidia-smi

# 파이토치/쿠다
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# tmux
tmux new -s train
tmux attach -t train

# Jupyter
jupyter notebook --no-browser --port 8888
# (로컬) ssh -L 8888:localhost:8888 ...

# S3 동기화
aws s3 sync ~/checkpoints s3://<bucket>/checkpoints
```

---

## 15) 문제가 생기면 (빠른 진단)

- **vCPU limit 에러**: Service Quotas → G and VT vCPU 0 → **Increase 요청**  
- **SSH 불가**: 보안그룹 22/TCP My IP 인지 확인, 키 권한 `chmod 400`  
- **GPU 인식 X**: DLAMI 사용 여부 확인, `nvidia-smi` 출력 확인  
- **비용 급증**: On-Demand로 켰는지 확인, Billing Alarm/Cost Explorer 점검

---

## 부록 A. 권장 사양 요약

| 항목 | 값 |
| --- | --- |
| 리전 | `us-east-1` |
| 인스턴스 | `g4dn.xlarge` (T4 16GB) |
| 구매옵션 | **Spot** |
| 스토리지 | 100~200GB, `gp3` |
| AMI | **Deep Learning AMI (Ubuntu 22.04, PyTorch 포함)** |
| 보안그룹 | SSH(22) My IP / Jupyter(8888) My IP |
| 한도 | G and VT vCPU = 8 |

---

## 부록 B. 해커톤 운영 팁

- **데이터/코드 버전관리**: `git` + `dvc` 또는 S3 버킷
- **체크포인트 전략**: `best.pt` + `last.pt`+ N-step rolling
- **실험 로그**: `tensorboard` 또는 `wandb` (로컬 포트 포워딩)
- **성능/비용 밸런스**: 증강/배치/정밀도(AMP)로 속도 개선 → 비용 절감
  ```python
  # PyTorch AMP 예시
  scaler = torch.cuda.amp.GradScaler()
  with torch.cuda.amp.autocast():
      loss = criterion(model(inputs), targets)
  scaler.scale(loss).backward()
  scaler.step(optimizer); scaler.update()
  ```

---

이 문서대로만 진행하면, **8일/100달러 내**에서 안정적으로 학습을 돌릴 수 있다.  
이후엔 **Docker 이미지**로 환경을 고정하거나, **SageMaker**로 확장하는 것도 고려하자.
