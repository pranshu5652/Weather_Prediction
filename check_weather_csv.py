
from pathlib import Path
import pandas as pd
from io import StringIO

file_path = Path("data/open-meteo-26.82N80.93E127m.csv")

with open(file_path, "r", encoding="utf-8-sig") as f:
    lines = f.readlines()

headers = [
    i for i, line in enumerate(lines)
    if line.strip().startswith("time,")
]

print("Header positions:", headers)

if len(headers) < 2:
    raise ValueError("CSV mein hourly/daily ke dono headers nahi mile.")

daily_text = "".join(lines[headers[1]:])
daily = pd.read_csv(StringIO(daily_text))

print("\nDaily columns:")
print(daily.columns.tolist())

print("\nLatest daily records:")
print(daily.tail(10).to_string(index=False))
