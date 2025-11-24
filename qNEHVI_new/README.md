# qNEHVI - Multi-objective Bayesian Optimization with Auto Variance Detection

## 개요
qNEHVI (q-Noisy Expected Hypervolume Improvement) 기반 다목적 베이지안 최적화 코드입니다.

## 주요 특징

✅ **자동 분산 감지**: 반복 측정 데이터의 분산을 자동으로 계산하여 노이즈 처리

✅ **반복 측정 지원**: 각 샘플별로 다른 측정 횟수 지원

✅ **FixedNoiseGP 모델**: 측정 노이즈를 고려한 Gaussian Process 모델

✅ **Hypervolume 계산**: Pareto front의 품질 평가

## 분산 처리 방식

1. **반복 측정 샘플**: 각 샘플의 분산을 개별적으로 계산하여 사용
2. **단일 측정 샘플**: 반복 측정된 샘플들의 평균 분산을 사용
3. **모든 샘플이 단일 측정**: default_noise (0.0001) 사용

## 사용 방법

코드를 실행하기 전에 `x_data`와 `y_data`를 정의해야 합니다:

```python
# 예시: x_data와 y_data 정의
x_data = [...]  # 입력 변수 데이터
y_data = [...]  # 출력 목적함수 데이터 (반복 측정 포함 가능)
```

## 실전 사용 권장

- **이 코드는 반복 측정 데이터가 있을 때 사용**하세요
- 모든 샘플이 한 번씩만 측정된 경우에는 **'qNEHVI_MOBO_code_GPU_한번씩 시행하는 코드'**를 사용하세요

## 요구사항

- PyTorch
- BoTorch
- GPyTorch
- NumPy
- Pandas
