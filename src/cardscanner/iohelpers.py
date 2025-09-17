import pathlib
import json
import csv

def append_rows_csv(csv_path: pathlib.Path, rows: list, header: list):
    """Append rows to CSV, writing header only if the file doesn’t exist yet."""
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerows(rows)

def merge_json(json_path: pathlib.Path, new_dict: dict):
    """Merge dict into JSON-on-disk (create if missing)."""
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                cur = json.load(f)
            except Exception:
                cur = {}
    else:
        cur = {}
    cur.update(new_dict)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)