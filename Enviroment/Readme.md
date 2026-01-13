# YOLO11n TensorRT (FP16) on Jetson Orin Nano

Jetson Orin Nano (JetPack 6.x / CUDA 12.6) 환경에서 YOLO11n 모델을 TensorRT FP16 엔진으로 변환하고 실행하는 가이드입니다.

## 환경 사양

| 항목 | 내용 |
|------|------|
| Device | Jetson Orin Nano |
| JetPack | 6.x (L4T R36) |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| Python | 3.10.12 |
| PyTorch | 2.5.0a0 (NVIDIA build, CUDA enabled) |
| torchvision | 0.20.0 (source build, CUDA enabled) |

## 1. 시스템 확인

### CUDA 확인
```bash
nvcc --version
# 출력: Cuda compilation tools, release 12.6, V12.6.68
```

### GPU 상태 확인
```bash
sudo tegrastats  # Jetson 권장
```

## 2. PyTorch (CUDA Enabled) 설치

⚠️ Jetson에서는 `pip install torch`로 설치되는 PyTorch가 CPU 전용이므로 NVIDIA JetPack 전용 wheel을 사용해야 합니다.

```bash
# 기존 torch 제거
python3 -m pip uninstall -y torch torchvision torchaudio

# NVIDIA PyTorch wheel 설치 (JetPack 6.x / Python 3.10)
python3 -m pip install \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

### CUDA PyTorch 확인
```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## 3. cuSPARSELt 설치

PyTorch 2.4+ 빌드는 cuSPARSELt 런타임을 별도로 요구합니다.

```bash
# cuSPARSELt 다운로드 (CUDA 12.x, aarch64)
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive.tar.xz

# 압축 해제 및 설치
tar -xvf libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive.tar.xz
sudo cp libcusparseLt.so* /usr/lib/aarch64-linux-gnu/
sudo ldconfig
```

⚠️ `ldconfig: libcusparseLt.so.0 is not a symbolic link` 경고는 무시해도 됩니다.

## 4. torchvision 빌드 (CUDA Extension 포함)

### 빌드 의존성 설치
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git \
  libjpeg-dev zlib1g-dev libpng-dev \
  python3-dev python3-opencv
```

### numpy 버전 고정
```bash
python3 -m pip install "numpy==1.26.1"
```

### torchvision 소스 빌드
```bash
git clone https://github.com/pytorch/vision torchvision
cd torchvision
git checkout v0.20.0
```

### 환경 변수 설정 (Orin Nano = SM 8.7)
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.7"
export FORCE_CUDA="1"
```

### 빌드 및 설치
```bash
python3 -m pip uninstall -y torchvision || true
python3 -m pip install -v --no-cache-dir --no-build-isolation .
```

### 확인
```python
import torch, torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())
```

## 5. Ultralytics (YOLO11) 설치

⚠️ Jetson에서는 `opencv-python` (pip) 대신 `python3-opencv` (apt) 사용을 권장합니다.

```bash
python3 -m pip install ultralytics --no-deps
yolo checks
```

## 6. ONNX Export Dependencies

```bash
python3 -m pip install "onnx>=1.12,<2" "onnxslim>=0.1.71"
```

## 7. YOLO11n → TensorRT FP16 Engine Export

```bash
yolo export model=yolo11n.pt format=engine half=True device=0 imgsz=640
```

### Export 결과
- ✅ ONNX export 성공
- ✅ TensorRT build 성공
- ✅ FP16 engine 생성 완료

### 생성된 파일 확인
```bash
ls -lh *.engine *.onnx
# yolo11n.onnx (~11 MB)
# yolo11n.engine (~8.4 MB)
```

## 8. TensorRT Engine Inference

### 이미지 추론
```bash
yolo predict model=yolo11n.engine source=sample.jpg device=0
```

### 실시간 카메라
```bash
yolo predict model=yolo11n.engine source=0 device=0
```

## 9. Performance Optimization (선택사항)

### 전력/클럭 고정
```bash
sudo nvpmodel -m 2
sudo jetson_clocks
```

### 실시간 모니터링
```bash
sudo tegrastats
```

## 최종 상태

✅ PyTorch CUDA enabled  
✅ torchvision CUDA extension enabled  
✅ ONNX export successful  
✅ TensorRT FP16 engine generated  
✅ Real-time inference on Jetson Orin Nano  

🚀 **Environment is ready for real-time Edge AI deployment**
