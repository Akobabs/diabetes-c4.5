"""Recreate the CSV from the preserved OpenML ARFF; never cleans or imputes values."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/raw/pima_diabetes.arff"
metadata = json.loads((ROOT / "data/raw/pima_openml_metadata.json").read_text())
if hashlib.md5(source.read_bytes()).hexdigest() != metadata["data_set_description"]["md5_checksum"]:
    raise ValueError("OpenML source checksum mismatch")
content = source.read_text().split("@data", 1)[1]
rows = []
for line in content.splitlines():
    if not line.strip() or line.lstrip().startswith("%"):
        continue
    cells = line.strip().split(",")
    if len(cells) != 9:
        raise ValueError("Expected nine ARFF fields")
    for cell in cells[:8]:
        float(cell)
    cells[-1] = {"tested_negative": "0", "tested_positive": "1"}[cells[-1].strip()]
    rows.append(",".join(cells))
if len(rows) != 768 or len(set(rows)) != 768:
    raise ValueError("Expected 768 distinct Pima records")
header = "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome"
target = ROOT / "data/derived/pima_diabetes.csv"
target.parent.mkdir(parents=True, exist_ok=True)
# Fixed CRLF preserves the original converted file's documented checksum.
target.write_bytes(("\r\n".join([header] + rows) + "\r\n").encode("utf-8"))
expected_csv_sha256 = "d765aa828a47e8d3ba2f4de925891be77803fd0e9568d7c213cdf10c8e83d0b2"
if hashlib.sha256(target.read_bytes()).hexdigest() != expected_csv_sha256:
    raise ValueError("Converted CSV does not match the documented reference checksum")
print(f"Verified and converted {len(rows)} rows: {target}")
