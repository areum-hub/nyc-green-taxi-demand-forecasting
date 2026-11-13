"""NYC TLC Green Taxi 전처리: 2025년 독립 재현 교차검증 전용.

실행 전 NYC TLC 2024~2025년 Parquet 24개를 data/raw/에 저장합니다.
2026년 1~6월 자료가 있으면 품질 집계에만 포함하며 모델 학습에는 쓰지 않습니다.

이 파일은 4Sight 팀의 2026년 최종 학습 코드가 아닌 별도 재현 코드입니다.
"""
from pathlib import Path
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
METRICS = ROOT / "reports" / "metrics"
PROCESSED.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

SOURCE_COLUMNS = [
    "lpep_pickup_datetime", "lpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance", "fare_amount",
]
NAMES = ["pickup", "dropoff", "pickup_zone", "dropoff_zone", "distance", "fare"]


def main():
    frames, quality = [], []
    start = time.monotonic()
    for year in [2024, 2025, 2026]:
        for month in range(1, 7 if year == 2026 else 13):
            files = sorted(RAW.glob(f"green_tripdata_{year}-{month:02d}*.parquet"))
            if not files:
                if year == 2026:
                    continue  # 2026년은 선택적 품질 확인 자료
                raise FileNotFoundError(
                    f"Missing green_tripdata_{year}-{month:02d}.parquet in {RAW}"
                )
            if len(files) > 1:
                raise ValueError(f"Duplicate monthly input files: {files}")
            # Windows에서도 동작하는 표준 PyArrow 엔진 이용
            trips = pd.read_parquet(files[0], columns=SOURCE_COLUMNS, engine="pyarrow")
            trips.columns = NAMES
            for col in ["pickup", "dropoff"]:
                trips[col] = pd.to_datetime(trips[col], errors="coerce")
            in_month = trips.pickup.dt.year.eq(year) & trips.pickup.dt.month.eq(month)
            valid_zone = trips.pickup_zone.between(1, 263)
            mask = in_month & valid_zone
            quality.append({
                "month": f"{year}-{month:02d}",
                "raw": len(trips),
                "wrong_month": int((~in_month).sum()),
                "bad_zone_remaining": int((in_month & ~valid_zone).sum()),
                "valid": int(mask.sum()),
            })
            frames.append(trips.loc[mask].copy())
            print(f"{year}-{month:02d}: raw={len(trips):,}, valid={mask.sum():,} "
                  f"elapsed={time.monotonic()-start:.1f}s", flush=True)

    if not frames:
        raise RuntimeError("No TLC trip records found.")
    clean = pd.concat(frames, ignore_index=True)
    clean.to_pickle(PROCESSED / "valid_trips.pkl")
    pd.DataFrame(quality).to_csv(METRICS / "raw_quality_reproduced.csv", index=False)
    print(f"Cleaned {len(clean):,} records; "
          f"2025 CV will exclude 2026 records in run_4fold_cv.py.")


if __name__ == "__main__":
    main()
