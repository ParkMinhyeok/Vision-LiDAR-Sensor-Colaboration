# YOLO11n TensorRT (FP16) on Jetson Orin Nano

이 문서는 **Jetson Orin Nano (JetPack 6.x / CUDA 12.6)** 환경에서  
**YOLO11n 모델을 TensorRT FP16 엔진으로 변환하고 실행하기까지의 전체 과정**을 정리합니다.

> ✅ 최종 결과: `yolo11n.engine` 생성 및 GPU 추론 성공

---

## 1. Environment Overview

| 항목 | 내용 |
|---|---|
| Device | Jetson Orin Nano |
| JetPack | 6.x (L4T R36) |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| Python | 3.10.12 |
| PyTorch | 2.5.0a0 (NVIDIA build, CUDA enabled) |
| torchvision | 0.20.0 (source build, CUDA enabled) |

---

## 2. System Check

### CUDA 확인
```bash
nvcc --version
