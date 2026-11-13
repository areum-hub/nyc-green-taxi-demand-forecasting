# NYC Green Taxi Demand Forecasting

**LightGBM Poisson 기반 지역별 승차 건수 예측 | 4Sight 팀 프로젝트 포트폴리오**

NYC Green Taxi 운행 데이터를 활용해 **지역별 다음 1시간의 관측 승차 건수**를 예측하고, 그 결과를 가상 차량 배치와 영업 효율화 의사결정에 활용하는 프로젝트입니다. 예측 정확도뿐 아니라 **시간순서에 따른 검증, 기준선 비교, 지역별 수요 분포의 예측 품질**을 함께 살펴봤습니다.

- **데이터:** NYC TLC Green Taxi Trip Records, 2024년 1월~2026년 6월(월별 Parquet 30개)
- **분석 단위:** 주요 20개 승차 지역 × 시간
- **모델:** LightGBM Poisson 회귀
- **활용 목표:** 예상 승차 비율을 활용한 가상 차량 배치, 영업 효율화 대시보드 및 지도 시각화

## 프로젝트 기여도

| 구분 | 기여 및 산출물 |
| --- | --- |
| **개인 기여 — 고아름** | 승차 건수가 0 이상의 건수형 데이터(Count Data)라는 점에 착안해 **LightGBM Poisson 목적함수 적용을 제안**하고, **모델을 직접 구현**했습니다. |
| **4Sight 팀 공동 산출물** | 데이터 전처리·분석과 최종모형 평가, **가상 차량 배치 시나리오 구상**, **영업 효율화 대시보드 및 지도 시각화 제작**, 최종 발표를 함께 수행했습니다. |

## 핵심 평가 결과

| 평가 구분 | 지표 | LightGBM Poisson | 기준선(Baseline) |
| --- | --- | ---: | ---: |
| **2025년 독립 재현 4-Fold** | MAE 평균 ± Fold 표준편차 | **1.141 ± 0.064** | 과거 3주 평균 1.278 ± 0.023 |
| 2025년 독립 재현 4-Fold | TVD 평균 | **0.293** | 요일·시간대 평균 0.303 |
| **2026년 상반기 팀 최종모형** | MAE | **1.034** | 과거 3주 평균 1.129 |
| 2026년 상반기 팀 최종모형 | RMSE | **1.918** | 과거 3주 평균 2.136 |
| 2026년 상반기 팀 최종모형 | TVD | **0.321** | 요일·시간대 평균 0.327 |
| 2026년 상반기 팀 최종모형 | Top-5 승차 포착 비율 | **68.29%** | 요일·시간대 평균 67.34% |

2026년 팀 최종모형의 MAE는 과거 3주 평균보다 **약 8.5% 낮았으나**, 승차 비율 기반 TVD와 Top-5 지표의 개선은 제한적이었습니다.

> **평가 출처 구분:** 2025년 4-Fold는 별도로 구성한 **독립 재현 실험**이고, 2026년 상반기는 **4Sight 팀이 보고한 최종모형 평가 결과**입니다. 같은 구현을 연속 검증한 수치로 직접 비교하지 않습니다.

## 평가 결과 시각화

### 2025년 독립 재현: 4-Fold MAE 비교

![2025년 Fold별 MAE 비교](reports/figures/2025_cv_mae_by_fold.svg)

[MAE 평균 ± Fold 간 표준편차 그래프](reports/figures/2025_cv_mae_mean_sd.svg)

### 2026년 팀 최종모형: 기준선 비교

![2026년 상반기 모델과 기준선 MAE 비교](reports/figures/2026_holdout_mae.svg)

[TVD 및 Top-5 승차 포착 비율 비교 그래프](reports/figures/2026_share_metrics.svg)

## 예측 결과의 운영 활용

팀에서는 예측 승차 비율을 바탕으로 **20대의 차량을 가상 배분**하는 시나리오를 구상하고, 이를 영업 효율화 대시보드와 지도 시각화로 표현했습니다.

예를 들어 **2026년 6월 1일 17시**의 예측 비율에 따른 배분 사례는 East Harlem North **28.7% → 6대**, East Harlem South **26.3% → 5대**입니다.

또한 **2026년 6월 1일 07시** East Harlem North에서는 실제 승차 **53건**에 비해 예측은 **약 28.5건**으로, 혼잡 시간대의 과소예측을 확인할 수 있습니다. [해당 날짜의 시간별 실제·예측 데이터](reports/metrics/2026_east_harlem_north_hourly.csv)

이 배치는 **실제 배차 실험이 아닌 가상 시나리오**입니다. TLC 데이터만으로는 미충족 수요나 실제 가용 차량을 알 수 없으므로, 시나리오의 Coverage Proxy 차이를 추가 승차 건수나 실현 이익으로 해석하지 않습니다.

## 독립 재현 코드 및 실행 방법

저장소에는 **2025년 Expanding Window 4-Fold 독립 재현**을 위한 전처리·학습·평가·시각화 코드를 공개했습니다.

| 단계 | 실행 파일 | 산출물 |
| --- | --- | --- |
| 1. 데이터 전처리 | [`src/prepare_data.py`](src/prepare_data.py) | 로컬 정제 데이터 및 [월별 품질 결과](reports/metrics/raw_quality_reproduced.csv) |
| 2. LightGBM 학습·평가 | [`src/run_4fold_cv.py`](src/run_4fold_cv.py) | [Fold별 지표](reports/metrics/2025_4fold_model_scores.csv), [평균·표준편차](reports/metrics/2025_4fold_mean_sd.csv) |
| 3. SVG 그래프 생성 | [`src/visualize_results.py`](src/visualize_results.py) | [시각화 파일](reports/figures/) |

**실행 환경:** Python 3.10 이상, Windows PowerShell

```powershell
git clone https://github.com/areum-hub/nyc-green-taxi-demand-forecasting.git
cd nyc-green-taxi-demand-forecasting
python -m pip install -r requirements.txt

# NYC TLC에서 2024~2025년 월별 파일 24개를 내려받아
# data/raw/green_tripdata_YYYY-MM.parquet 형식으로 저장
python src/prepare_data.py
python src/run_4fold_cv.py
python src/visualize_results.py
```

2026년 1~6월의 원본 파일 6개는 선택 사항이며, 전처리 시 품질 집계에는 포함할 수 있지만 **2025년 독립 재현 학습·검증에는 사용하지 않습니다**. 대용량 원본 데이터와 전처리 중간 파일은 GitHub에 포함하지 않습니다.

본 프로젝트는 별도의 API Key나 환경변수 설정 없이 실행할 수 있으므로 `.env.example` 파일을 사용하지 않습니다.

**학습·검증 경계:** Fold별 CSV의 `train_until_exclusive`는 학습에 포함되지 않는 첫 날짜입니다. 예를 들어 `2025-01-01`은 2024년 12월 31일까지 학습하고 2025년 1월 1일부터 검증한다는 뜻입니다.

> 이 공개 코드는 **2025년 독립 재현용**입니다. 2026년 팀 최종모형의 원본 학습·평가 파이프라인과 동일하지 않으며, 집계 결과를 시각화하는 코드만으로 팀 최종모형을 재현할 수는 없습니다.

**추가 자료:** [검증 방법론 및 지표 정의](docs/methodology.md) · [2026년 팀 평가 수치](reports/metrics/2026_team_holdout_summary.csv) · [독립 재현 실험 설정](reports/metrics/methodology.json)

**데이터 출처:** [NYC Taxi & Limousine Commission — Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
