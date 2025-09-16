#!/usr/bin/env python3
import os, re, csv, json, subprocess, pathlib
import cv2, numpy as np
from PIL import Image

# -----------------------------
# Photoshop configuration
# -----------------------------
PHOTOSHOP_BUNDLE_ID = "com.adobe.Photoshop"   # version-agnostic bundle ID for macOS
ACTION_SET = "Default Actions"                # Photoshop Action Set to run
ACTION_NAME = "Calibrate Scanned Image"       # Photoshop Action name inside that set
JPEG_QUALITY = 10                             # output quality (0–12 in Photoshop)

# -----------------------------
# OCR (Tesseract) configuration
# -----------------------------
USE_TESSERACT = True       # set to False if you want to skip OCR entirely
TESS_LANGS = "eng"         # language(s) for Tesseract; add +osd for orientation
ROTATE_MODE = "cw"         # extra 90° rotation per card: "cw" or "ccw"
DEFAULT_ROTATE_MODE = "cw" # default if user doesn’t specify
DPI_FOR_OCR = 300          # upsample images to simulate this DPI for OCR
BIN_THR_PCT = (5, 95)      # percentile stretch for contrast (clip darkest/lightest ends)

# -----------------------------
# Helper functions
# -----------------------------

# ---------- ROI-based OCR for dealer cards ----------

def _roi(img, x0, y0, x1, y1):
    """Crop by fractional coordinates (0..1)."""
    h, w = img.shape[:2]
    X0, Y0 = int(x0*w), int(y0*h)
    X1, Y1 = int(x1*w), int(y1*h)
    return img[Y0:Y1, X0:X1].copy()

def _prep_text(bgr, invert=False):
    """High-contrast binarization for handwriting/print."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # light stretch
    lo, hi = np.percentile(g, (5, 95))
    if hi > lo:
        g = np.clip((g - lo) * (255.0/(hi - lo)), 0, 255).astype(np.uint8)
    # adaptive threshold works well on the card stock
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 9)
    if invert:
        thr = 255 - thr
    return thr

def _tess(img_bin, psm=6, allow=None, lang=TESS_LANGS):
    import pytesseract
    config = f"--oem 1 --psm {psm}"
    if allow:
        # whitelist characters (handy for prices and years)
        config += f" -c tessedit_char_whitelist={allow}"
    return pytesseract.image_to_string(img_bin, config=config, lang=lang).strip()

def ocr_card_structured(img_path: pathlib.Path, debug=False, out_dir=None):
    """
    Read key fields from a processed card:
      - condition (MNH/MLH/MVLH or blank = Used)
      - year (4 digits, top-left)
      - scott (e.g., #MH287 or 123, A12a, etc.)
      - cat_price (e.g., $1.60)
      - sell_price (e.g., $1.00)
    Returns dict with fields + raw_texts.
    """
    bgr = cv2.imread(str(img_path))
    h, w = bgr.shape[:2]

    # --- ROIs (fractions of width/height) ---
    # Adjust if needed; these work for the examples you shared.
    ROIS = {
        "condition": (0.02, 0.01, 0.25, 0.11),   # top-left corner strip
        "year":      (0.02, 0.10, 0.25, 0.20),   # just below condition
        "scott":     (0.28, 0.02, 0.62, 0.14),   # centered top
        # right third; we'll read two prices from left->right
        "prices":    (0.62, 0.02, 0.98, 0.16),
    }

    crops = {k: _roi(bgr, *ROIS[k]) for k in ROIS}

    # Preprocess for OCR
    bin_condition = _prep_text(crops["condition"])
    bin_year      = _prep_text(crops["year"])
    bin_scott     = _prep_text(crops["scott"])
    bin_prices    = _prep_text(crops["prices"])

    # --- OCR with targeted settings ---
    cond_txt = _tess(bin_condition, psm=7)  # single line
    year_txt = _tess(bin_year,      psm=7, allow="0123456789")  # digits only
    scott_txt= _tess(bin_scott,     psm=7)  # single line, mixed chars
    prices_txt=_tess(bin_prices,    psm=6, allow="$0123456789.,")  # a couple of numbers

    # --- Parse/cleanup ---
    # condition normalization
    cond_norm = None
    for token in ["MNH", "MLH", "MVLH"]:
        if token in cond_txt.upper():
            cond_norm = token
            break
    if cond_norm is None and cond_txt.strip() == "":
        cond_norm = "Used"  # your convention when left blank

    # year: first 4-digit number that looks like a year 1800–2099
    year = None
    m = re.search(r"\b(18|19|20)\d{2}\b", year_txt)
    if m:
        year = m.group(0)

    # scott: accept formats like MH287, #123, A12a, 123-125 etc.
    scott = None
    m = re.search(r"(?:#\s*)?([A-Z]{0,3}\d+[A-Za-z]?(-\d+[A-Za-z]?)?)", scott_txt.strip())
    if m:
        scott = m.group(1)

    # prices: try to pull two amounts, left-to-right
    amounts = re.findall(r"\$?\s*\d+(?:[.,]\d{2})?", prices_txt)
    amounts = [a.replace(" ", "").replace(",", ".") for a in amounts]
    # normalize to $X.YY format if possible
    def norm_price(a):
        a = a.lstrip("$")
        if "." not in a:
            a = f"{a}.00"
        return f"${a}"
    cat_price = norm_price(amounts[0]) if len(amounts) >= 1 else None
    sell_price= norm_price(amounts[1]) if len(amounts) >= 2 else None

    if debug and out_dir:
        vis = bgr.copy()
        for k,(x0,y0,x1,y1) in ROIS.items():
            X0,Y0,X1,Y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
            cv2.rectangle(vis, (X0,Y0), (X1,Y1), (0,255,0), 3)
            cv2.putText(vis, k, (X0, max(0,Y0-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imwrite(str(pathlib.Path(out_dir)/f"{img_path.stem}_roi_debug.jpg"), vis)
        cv2.imwrite(str(pathlib.Path(out_dir)/f"{img_path.stem}_roi_prices_bin.png"), bin_prices)

    return {
        "condition_raw": cond_txt, "condition": cond_norm,
        "year_raw": year_txt,       "year": year,
        "scott_raw": scott_txt,     "scott": scott,
        "prices_raw": prices_txt,   "cat_price": cat_price, "sell_price": sell_price,
    }

def rotate_90(img_bgr, mode="cw"):
    """Rotate an image 90° clockwise or counter-clockwise."""
    if mode == "cw":
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    else:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

def contrast_stretch_gray(gray, lo_pct=5, hi_pct=95):
    """Stretch grayscale levels so lo_pct → black, hi_pct → white."""
    lo, hi = np.percentile(gray, (lo_pct, hi_pct))
    if hi <= lo:  # avoid division by zero
        return gray
    out = np.clip((gray - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return out

def prepare_for_ocr(bgr):
    """
    Convert image into a high-contrast, binarized version suitable for OCR.
    Steps:
      1. Convert to grayscale
      2. Contrast stretch
      3. Adaptive threshold
      4. Upscale if too small
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = contrast_stretch_gray(g, *BIN_THR_PCT)
    thr = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 9
    )
    h, w = thr.shape
    target_h = max(h, int(DPI_FOR_OCR / 300.0 * h))
    if target_h > h:  # upscale if smaller than our desired DPI
        scale = target_h / h
        thr = cv2.resize(thr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return thr

# -----------------------------
# Step 1: Detect and crop dealer cards
# -----------------------------

def detect_and_crop(src_path: str, out_dir: str, rotate_mode: str, debug=False):
    os.makedirs(out_dir, exist_ok=True)
    img  = cv2.imread(src_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    S    = min(H, W)

    # find starting index: look at existing files in out_dir
    existing = [f for f in os.listdir(out_dir) if f.startswith("card_") and f.endswith(".jpg")]
    numbers = []
    for f in existing:
        m = re.match(r"card_(\d+)_raw\.jpg", f)
        if m:
            numbers.append(int(m.group(1)))
    next_index = max(numbers)+1 if numbers else 1


    # Scale-aware hyperparameters (kernels scaled from image size)
    blk     = max(15, int(round(0.015 * S)) | 1)  # block size for adaptive threshold
    C       = -8                                  # brightness offset for threshold
    close_k = max(5, int(round(0.010 * S)))       # kernel for closing
    erode_k = max(7, int(round(0.015 * S)))       # kernel for erosion

    def pipeline(adapt_block=blk, adapt_C=C, close_sz=close_k, erode_sz=erode_k):
        """Build binary mask → close gaps → erode inward."""
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, adapt_block, adapt_C
        )
        clean = cv2.morphologyEx(thr, cv2.MORPH_CLOSE,
                                 np.ones((close_sz, close_sz), np.uint8), iterations=1)
        eroded = cv2.erode(clean,
                           np.ones((erode_sz, erode_sz), np.uint8), iterations=1)
        return thr, clean, eroded

    def find_cards(bin_img):
        """Find 4-cornered contours big enough to be cards."""
        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 0.02 * H * W:    # ignore tiny blobs (<2% of image)
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                boxes.append(approx)
        return sorted(boxes, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    # Try normal pipeline
    thr, clean, eroded = pipeline()
    cards = find_cards(eroded)

    # Fallbacks if detection is too strict
    if len(cards) < 2:
        # gentler erosion
        eroded2 = cv2.erode(clean, np.ones((max(5, erode_k//2), max(5, erode_k//2)), np.uint8), 1)
        cards = find_cards(eroded2)
        if debug: cv2.imwrite(os.path.join(out_dir, "_dbg_eroded2.png"), eroded2)

    if len(cards) < 2:
        # tweak threshold offset
        thr2, clean2, eroded3 = pipeline(adapt_block=blk, adapt_C=-4,
                                         close_sz=close_k, erode_sz=max(5, erode_k//2))
        cards = find_cards(eroded3)
        if debug:
            cv2.imwrite(os.path.join(out_dir, "_dbg_thr2.png"), thr2)
            cv2.imwrite(os.path.join(out_dir, "_dbg_eroded3.png"), eroded3)

    if len(cards) < 2:
        # last resort: Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        er4 = cv2.erode(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE,
                        np.ones((close_k, close_k), np.uint8)),
                        np.ones((max(5, erode_k//2), max(5, erode_k//2)), np.uint8), 1)
        cards = find_cards(er4)
        if debug:
            cv2.imwrite(os.path.join(out_dir, "_dbg_otsu.png"), otsu)
            cv2.imwrite(os.path.join(out_dir, "_dbg_er4.png"), er4)

    if debug:
        # Save masks and a visualization of contours
        cv2.imwrite(os.path.join(out_dir, "_dbg_thr.png"), thr)
        cv2.imwrite(os.path.join(out_dir, "_dbg_clean.png"), clean)
        cv2.imwrite(os.path.join(out_dir, "_dbg_eroded.png"), eroded)
        dbg = img.copy()
        for c in cards:
            cv2.drawContours(dbg, [c], -1, (0,255,0), 6)
        cv2.imwrite(os.path.join(out_dir, "_dbg_contours.jpg"), dbg)

    if not cards:
        raise RuntimeError("No dealer cards detected. Check _dbg_*.png in the output folder.")

    # Warp each detected card into a rectangle and rotate
    paths = []
    for i, poly in enumerate(cards, 0):
        rect = cv2.minAreaRect(poly)
        box  = cv2.boxPoints(rect).astype("float32")
        Wp, Hp = map(int, rect[1])
        dst = np.array([[0, Hp-1],[0,0],[Wp-1,0],[Wp-1,Hp-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(box, dst)
        warp = cv2.warpPerspective(img, M, (Wp, Hp))

        if warp.shape[1] > warp.shape[0]:
            warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
        warp = rotate_90(warp, rotate_mode)

        # use next_index instead of restarting at 1
        idx = next_index + i
        tmp = os.path.join(out_dir, f"card_{idx:04d}_raw.jpg")  # 0001, 0002, etc.
        Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)).save(tmp, quality=95)
        paths.append(tmp)

    return paths
# -----------------------------
# Step 2: Run Photoshop Action
# -----------------------------
def run_ps_action(in_path: pathlib.Path, out_path: pathlib.Path):
    """
    Open a file in Photoshop, run a predefined Action, save result as JPEG.
    Uses AppleScript + Photoshop's JavaScript engine.
    """
    in_posix = in_path.resolve().as_posix()
    out_posix = out_path.resolve().as_posix()
    out_dir = out_path.parent.resolve().as_posix()
    ascript = f"""
    do shell script "mkdir -p " & quoted form of "{out_dir}"
    tell application id "{PHOTOSHOP_BUNDLE_ID}"
        activate
        set jsOpen to "var f=new File('{in_posix}'); if(!f.exists) 'ERROR: missing ' + f.fsName; else{{app.open(f);'OPEN_OK';}}"
        set rOpen to do javascript jsOpen
        if rOpen does not start with "OPEN_OK" then error rOpen
        set jsAct to "app.displayDialogs=DialogModes.NO; try{{app.doAction('{ACTION_NAME}','{ACTION_SET}');'ACT_OK';}}catch(e){{'ERROR: '+e;}}"
        set rAct to do javascript jsAct
        if rAct does not start with "ACT_OK" then error rAct
        set jsSave to "var f=new File('{out_posix}'); var o=new JPEGSaveOptions(); o.quality={JPEG_QUALITY}; app.activeDocument.saveAs(f,o,true);'SAVE_OK';"
        set rSave to do javascript jsSave
        if rSave does not start with "SAVE_OK" then error rSave
        close current document saving no
    end tell
    """
    subprocess.run(["osascript", "-e", ascript], check=True)

# -----------------------------
# Step 3: OCR each processed card
# -----------------------------
def ocr_card(img_path: pathlib.Path):
    """
    Run Tesseract OCR on an image, and try to parse price + Scott numbers.
    """
    if not USE_TESSERACT:
        return {"text": "", "price": None, "scott": []}
    import pytesseract

    bgr = cv2.imread(str(img_path))
    proc = prepare_for_ocr(bgr)
    config = "--oem 1 --psm 6 -l " + TESS_LANGS
    txt = pytesseract.image_to_string(proc, config=config)

    # Extract a price (simple regex)
    price = None
    m = re.search(r"(\$?\s*\d{1,3}(?:[,\d]{0,3})*(?:\.\d{1,2})?)\s*(?:USD)?", txt, re.I)
    if m:
        price = m.group(1).strip()

    # Extract Scott numbers (various formats)
    scotts = re.findall(r"(?:Scott\s*)?(?:#\s*)?([A-Z]?\d+[A-Za-z]?(?:-\d+[A-Za-z]?)?)", txt)

    return {"text": txt.strip(), "price": price, "scott": scotts}

# -----------------------------
# Orchestration
# -----------------------------
def process_scan(scan_path: str, out_dir: str, rotate_mode: str):
    """
    Full pipeline:
      1. Detect and crop dealer cards
      2. Run Photoshop Action on each
      3. OCR and save results to CSV/JSON
    """
    scan_path = pathlib.Path(scan_path)
    out_dir = pathlib.Path(out_dir)
    raw_paths = detect_and_crop(str(scan_path), str(out_dir), rotate_mode)

    final_paths = []
    for p in raw_paths:
        p = pathlib.Path(p)
        out_final = out_dir / f"{p.stem}_PS{p.suffix}"
        run_ps_action(p, out_final)
        final_paths.append(out_final)

    # OCR → build rows/json (no file I/O here)
    rows = []
    json_dump = {}
    for outp in final_paths:
        meta = ocr_card_structured(outp, debug=False, out_dir=str(out_dir))  # <-- use ROI OCR
        rows.append({
            "file": outp.name,
            "condition": meta.get("condition"),
            "year": meta.get("year"),
            "scott": meta.get("scott"),
            "cat_price": meta.get("cat_price"),
            "sell_price": meta.get("sell_price"),
        })
        json_dump[outp.name] = meta

    return final_paths, rows, json_dump

def iter_images(path: pathlib.Path, recursive: bool = False):
    """Yield absolute paths to JPG/JPEG/TIFF/PNG under path (file or folder)."""
    exts = {".jpg", ".jpeg", ".tiff", ".png", ".JPG", ".JPEG", ".TIFF", ".PNG"}
    path = path.resolve()
    if path.is_file():
        if path.suffix in exts:
            yield path
        return
    if recursive:
        for p in sorted(path.rglob("*")):
            if p.suffix in exts:
                yield p
    else:
        for p in sorted(path.glob("*")):
            if p.suffix in exts:
                yield p

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

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Dealer card pipeline: crop → rotate → Photoshop → OCR (file or folder)"
    )
    ap.add_argument("scan", help="input scan file OR a folder containing images")
    ap.add_argument("-o", "--out", default="cards_out", help="output folder")
    ap.add_argument("--rotate", choices=["cw", "ccw"], default=DEFAULT_ROTATE_MODE,
                    help="extra 90° rotation direction")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="recurse into subfolders when scan is a directory")
    args = ap.parse_args()

    if args.no_ocr:
        USE_TESSERACT = False

    in_path = pathlib.Path(args.scan).resolve()
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators across all inputs
    all_rows = []
    all_json = {}

    inputs = list(iter_images(in_path, recursive=args.recursive))
    if not inputs:
        raise SystemExit(f"No images found at: {in_path}")

    for src in inputs:
        print(f"→ Processing {src.name}")
        finals, rows, jdump = process_scan(str(src), str(out_dir), args.rotate)
        # Collect
        all_rows.extend(rows)
        all_json.update(jdump)

    # Append rows to CSV and merge JSON
    csv_path = out_dir / "cards_ocr.csv"
    json_path = out_dir / "cards_ocr.json"

    if USE_TESSERACT:
        header = ["file", "condition", "year", "scott", "cat_price", "sell_price"]
        append_rows_csv(csv_path, all_rows, header)
        merge_json(json_path, all_json)
        print(f"Wrote/updated: {csv_path} and {json_path}")

    print("Done.")