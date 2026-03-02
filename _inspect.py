import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

xl = pd.ExcelFile("data/Gen_AI_Dataset.xlsx")
print("Sheets:", xl.sheet_names)

for s in xl.sheet_names:
    print(f"\n=== {s} ===")
    df = xl.parse(s)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    # just show the query column and first few chars
    for i, row in df.iterrows():
        vals = [str(v)[:120] for v in row.values]
        print(f"  Row {i}: {vals}")
