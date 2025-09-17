#!/usr/bin/env python3
import argparse, pathlib
from .config import DEFAULT_ROTATE_MODE, USE_TESSERACT
from .pipeline import process_scan
from .imaging import iter_images
from .iohelpers import append_rows_csv, merge_json

def main():
    ap = argparse.ArgumentParser(description="Dealer card pipeline")
    ap.add_argument("scan", help="input scan file OR a folder")
    ap.add_argument("-o", "--out", default="./Stamps/Slices", help="output folder")
    ap.add_argument("--rotate", choices=["cw","ccw"], default=DEFAULT_ROTATE_MODE)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("-r", "--recursive", action="store_true")
    args = ap.parse_args()

    if args.no_ocr:
        # simple runtime override
        import src.cardscanner.config as cfg
        cfg.USE_TESSERACT = False

    in_path = pathlib.Path(args.scan).resolve()
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, all_json = [], {}
    inputs = list(iter_images(in_path, recursive=args.recursive))
    if not inputs:
        raise SystemExit(f"No images found at: {in_path}")

    for src in inputs:
        print(f"→ Processing {src.name}")
        finals, rows, jdump = process_scan(str(src), str(out_dir), args.rotate)
        all_rows.extend(rows); all_json.update(jdump)

    if USE_TESSERACT:
        header = ["file","condition","year","scott","cat_price","sell_price"]
        append_rows_csv(out_dir/"cards_ocr.csv", all_rows, header)
        merge_json(out_dir/"cards_ocr.json", all_json)
        print(f"Wrote/updated: {out_dir/'cards_ocr.csv'} and {out_dir/'cards_ocr.json'}")
    print("Done.")

if __name__ == "__main__":
    main()
