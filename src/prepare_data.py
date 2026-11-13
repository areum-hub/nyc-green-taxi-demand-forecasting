"""Preprocess local TLC Green Taxi Parquet for the independent 2025 CV.

Requires 2024-2025 monthly files under data/raw/. Optional 2026 Jan-Jun
files contribute only to the quality summary, not to the CV model.
"""
import sys
from pathlib import Path
from parquet_min import read_parquet_columns
import numpy as np
import pandas as pd
import time
repo=Path(__file__).resolve().parents[1]
root=repo/'data'/'raw'
out=repo/'data'/'processed'
out.mkdir(parents=True,exist_ok=True)
metrics=repo/'reports'/'metrics'
metrics.mkdir(parents=True,exist_ok=True)
columns=['lpep_pickup_datetime','lpep_dropoff_datetime','PULocationID','DOLocationID','trip_distance','fare_amount']
frames=[]
quality=[]
start=time.monotonic()
for y in [2024,2025,2026]:
 for m in range(1,13 if y!=2026 else 7):
  candidates=sorted(root.glob(f'green_tripdata_{y}-{m:02d}*.parquet'))
  if not candidates:
   if y==2026: continue
   raise FileNotFoundError(f'Missing {y}-{m:02d} Green Taxi Parquet in {root}. See README.md')
  if len(candidates)>1:
   raise ValueError(f"Duplicate monthly files for {y}-{m:02d}: {candidates}")
  fn=candidates[0]
  try:
   d=pd.read_parquet(fn,columns=columns)
  except (ImportError, ModuleNotFoundError):
   d=pd.DataFrame(read_parquet_columns(fn,columns))
  d=d.rename(columns=dict(zip(columns,['pickup','dropoff','pickup_zone','dropoff_zone','distance','fare'])))
  for c in ['pickup','dropoff']:
   if pd.api.types.is_datetime64_any_dtype(d[c]):
    d[c]=pd.to_datetime(d[c],errors='coerce')
   else:
    d[c]=pd.to_datetime(d[c],unit='us',errors='coerce')
  mask=d.pickup.dt.year.eq(y)&d.pickup.dt.month.eq(m)
  bad_month=int((~mask).sum())
  bad_zone=int((~d.loc[mask,'pickup_zone'].between(1,263)).sum())
  mask &= d.pickup_zone.between(1,263)
  quality.append({'month':f'{y}-{m:02d}','raw':len(d),'wrong_month':bad_month,'bad_zone_remaining':bad_zone,'valid':int(mask.sum())})
  frames.append(d.loc[mask])
  print(f'{y}-{m:02d} raw={len(d):,} valid={mask.sum():,} elapsed={time.monotonic()-start:.1f}s',flush=True)
frame=pd.concat(frames,ignore_index=True)
frame.to_pickle(out/'valid_trips.pkl')
pd.DataFrame(quality).to_csv(metrics/'raw_quality_reproduced.csv',index=False)
print('VALID TOTAL',len(frame),'byyear',frame.groupby(frame.pickup.dt.year).size().to_dict(),flush=True)
print('QUALITY',pd.DataFrame(quality).sum(numeric_only=True).to_dict(),flush=True)
