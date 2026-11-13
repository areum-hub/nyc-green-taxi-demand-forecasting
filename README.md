# NYC Green Taxi Demand Forecasting

**시계열 머신러닝 기반 뉴욕 그린 택시 승차 건수 예측 및 모델 성능 검증**

NYC TLC Green Taxi 운행 데이터를 활용해 **지역별 다음 1시간의 관측 승차 건수**를 예측하고, LightGBM Poisson 모델의 성능을 시계열 교차검증(Time-Series Cross Validation), 기준선(Baseline) 비교, 오차 분석과 함께 평가한 포트폴리오입니다. 지역별 승차 비율은 **가상 20대 차량 배분 시나리오**에도 활용합니다.

## 프로젝트 핵심 결과

| 평가 구분 | 평가지표 | LightGBM Poisson | 비교 기준선 |
|---|---|---:|---:|
| **2025년 독립 재현 4-Fold** | MAE 평균 ± Fold 간 표준편차 | **1.141 ± 0.064** | 과거 3주 평균: 1.278 ± 0.023 |
| **2026년 상반기 팀 최종모형 회고 평가** | MAE | **1.034** | 과거 3주 평균: 1.129 |
| 2026년 상반기 팀 최종모형 | RMSE | **1.918** | 과거 3주 평균: 2.136 |
| 2026년 상반기 팀 최종모형 | TVD | **0.321** | 요일·시간대 평균: 0.327 |
| 2026년 상반기 팀 최종모형 | Top-5 승차 포착 비율 | **68.29%** | 요일·시간대 평균: 67.34% |

> **검증 범위 구분:** 2025년 4-Fold는 팀 최종모형 코드를 그대로 재실행한 것이 아니라 별도로 구현한 **독립 재현 실험**입니다. 2026년 수치는 팀 프로젝트의 최종모형 평가 결과입니다. 두 수치를 동일한 학습 파이프라인의 전후 성능처럼 해석하지 않습니다.

## Python 시각화 결과

GitHub에서 바로 열어볼 수 있도록 평가 수치의 **SVG 그래프**를 공개했습니다. [Python 생성 코드](src/visualize_results.py)는 같은 CSV를 입력받아 시각화를 생성합니다. 아래의 SVG는 저장된 집계 수치를 기반으로 제작한 게시용 그림으로, Python 스크립트의 출력 파일과 시각적 스타일이 완전히 동일하지 않을 수 있습니다.

### 1. Fold별 MAE 변화 — 2025년 독립 재현
![2025 Fold별 MAE](reports/figures/2025_cv_mae_by_fold.svg)

### 2. MAE 평균과 표준편차 — 2025년 독립 재현
![2025 MAE 평균과 Fold 간 표준편차](reports/figures/2025_cv_mae_mean_sd.svg)

### 3. 2026년 상반기 최종모형 MAE 비교
![2026 최종모형 MAE](reports/figures/2026_holdout_mae.svg)

### 4. 승차 비율 예측 품질 비교
![2026 TVD 및 Top-5 비율](reports/figures/2026_share_metrics.svg)

### 시각화 재현

```bash
pip install pandas matplotlib
python src/visualize_results.py
```

실행 시 `reports/metrics/`의 집계 CSV를 읽고 `reports/figures/`에 SVG 그래프를 생성합니다. 이 코드는 **학습 모델을 재훈련하지 않고**, 이미 계산된 평가 지표를 시각화합니다.

## 모델 평가 자료

- [모델 검증 보고서 및 오차 분석](reports/readme.md)
- [검증 방법론과 한계](docs/methodology.md)
- [2025년 4-Fold별 평가 CSV](reports/metrics/2025_4fold_model_scores.csv)
- [2025년 평균·표준편차 CSV](reports/metrics/2025_4fold_mean_sd.csv)
- [2026년 팀 보고 결과 CSV](reports/metrics/2026_team_holdout_summary.csv)

## 데이터와 모델링

- **데이터:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), 2024년 1월~2026년 6월, 월별 Parquet 30개
- **원본 데이터 품질 검토:** 1,506,768건 중 승차 월·지역 코드 등을 확인해 1,501,696건을 유효 기록으로 정리(팀 프로젝트 기준)
- **목표 변수:** 지역별 다음 1시간의 실제 관측 승차 건수
- **예측 지역:** 과거 자료에서 선정한 상위 20개 승차 지역
- **모델:** LightGBM Poisson 및 시차(Lag)·달력·과거 완료 운행 흐름 특성

원본 Parquet는 파일 용량과 재현 절차를 고려해 GitHub에 업로드하지 않았습니다. 원본은 NYC TLC 사이트에서 내려받아 로컬에서 관리합니다.

## 비즈니스 해석과 한계

2026년 상반기 최종모형은 과거 3주 평균 기준선보다 MAE가 약 8.5% 낮았습니다. 다만 2026년 6월 1일 07시 East Harlem North의 실제 승차 53건을 약 28.5건으로 예측하는 등 **혼잡 시간대 과소예측** 문제가 남아 있습니다.

승차 비율에 따른 20대 배분은 가상 의사결정 실험이며, 실제 이용하지 못한 승객이나 가용 택시의 위치는 관측되지 않았습니다. 따라서 배분 시나리오의 지표를 실제 추가 승차 건수 또는 실현 이익으로 주장하지 않습니다.

## 프로젝트 출처 및 기여 구분

본 포트폴리오는 **4Sight 팀 프로젝트** 결과를 바탕으로 구성했습니다. 2026년 최종모형 결과는 팀 공동 산출물이며, 2025년 4-Fold 실험 및 포트폴리오 검증 자료는 별도로 재구성했습니다. 개별 기여와 팀 공동 산출물을 구분해 기술합니다.

**데이터 출처:** NYC Taxi & Limousine Commission (TLC).
