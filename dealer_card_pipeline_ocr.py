#!/usr/bin/env python3
import os
import re
import csv
import json
import subprocess
import pathlib
import cv2
import numpy as np
from PIL import Image
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# -----------------------------
# Config: choose OCR source
# -----------------------------
# "micro"  -> Tesseract on microcrops (recommended)
# "full"   -> Tesseract on whole header band via ROI (your current ocr_card_structured)
# "ai"     -> OpenAI JSON schema on microcrops (ai_read_from_microcrops)
OCR_SOURCE = "micro"             # "micro" | "full" | "ai"
AI_MODEL   = "gpt-4.1"           # used only if OCR_SOURCE == "ai"

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


# Map of common OCR misreads (letters to digits)
LOOKALIKE_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5",
    "Z": "2", "z": "2"
})

# This is the band where the header (Scott #, prices, condition) is expected
HEADER_BAND = (0.0, 0.0, 1.0, 0.2)
HEADER_LEFT = (0.00, 0.00, 0.18, 0.8)
HEADER_SCOTT = (0.18, 0.00, 0.53, 0.8)
HEADER_PRICE = (0.52, 0.00, 1.00, 1.0)

# inside price block:
PRICE_CAT    = (0.00, 0.00, 0.45, 1.00)
PRICE_SELL   = (0.45, 0.00, 1.00, 1.00)


SCHEMA = {
  "type": "object",
  "properties": {
    "condition": {"type":"string", "enum":["MNH","MLH","MVLH","Used",""]},
    "year":      {"type":"string"},      # "1976" or ""
    "scott":     {"type":"string"},      # "#681-682" or "MH239" or ""
    "cat_price": {"type":"string"},      # "$1.30" or ""
    "sell_price":{"type":"string"}       # "$1.00" or ""
  },
  "required": ["condition","year","scott","cat_price","sell_price"],
  "additionalProperties": False
}

def ai_read_from_microcrops(micro: dict, model="gpt-4o-mini"):
    """
    micro: {"left": Path, "scott": Path, "cat": Path, "sell": Path}
    Returns dict matching SCHEMA.
    """
    imgs = [
        _img_part(micro["left"]),
        _img_part(micro["scott"]),
        _img_part(micro["cat"]),
        _img_part(micro["sell"]),
    ]

    prompt = (
      "Extract five fields from these crops of a dealer card header:\n"
      "- left: condition (MNH/MLH/MVLH or blank=Used) and 4-digit year\n"
      "- scott: e.g., #681-682, 2341a, MH239\n"
      "- cat: catalog price like $1.30\n"
      "- sell: selling price like $1.00\n"
      "Return strictly the schema; use empty string if unreadable."
    )

    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a precise OCR parser. Output must match the schema exactly."},
                {"role":"user","content":[{"type":"text","text":prompt}, *imgs]}
            ],
            response_format={
                "type":"json_schema",
                "json_schema":{"name":"card_fields","schema":SCHEMA,"strict":True}
            }
        )
        meta = res.choices[0].message.parsed
        return meta
    except Exception as e:
        # Fallback: use plain JSON object if schema isn’t supported
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a precise OCR parser. Output valid JSON only."},
                {"role":"user","content":[{"type":"text","text":prompt}, *imgs]}
            ],
            response_format={"type":"json_object"}
        )
        raw = res.choices[0].message.content
        # be defensive if content comes back as a list
        if isinstance(raw, list):
            raw = "".join(part.get("text","") for part in raw if part.get("type")=="text")
        meta = json.loads(raw)
        # ensure keys exist (schema shape)
        for k in ("condition","year","scott","cat_price","sell_price"):
            meta.setdefault(k, "")
        return meta


# -----------------------------
# Helper functions
# -----------------------------
def _img_part(path: pathlib.Path):
    # pick a sensible MIME (JPEG default)
    ext = path.suffix.lower()
    mime = "image/png" if ext in {".png"} else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

def _clean_ascii(s: str) -> str:
    # strip curly quotes/dashes/odd glyphs Tesseract invents
    return re.sub(r"[^\w\-\s#$.,/]", "", s)

def _normalize_digits(s: str) -> str:
    # translate look-alikes only inside number-like runs
    def fix(m): return m.group(0).translate(LOOKALIKE_MAP)
    return re.sub(r"([$]?\s*[A-Za-z]*\d[\dOIlSsz.,]*)", fix, s)

def _extract_amount(txt: str):
    t = _normalize_digits(_clean_ascii(txt)).replace(" ", "").replace(",", ".")
    m = re.search(r"\$?\d+(?:\.\d{1,2})?", t)
    if not m: return None
    amt = m.group(0).lstrip("$")
    if "." not in amt: amt += ".00"
    return f"${amt}"

def _tess(img_bin, psm=6, allow=None, lang=TESS_LANGS, oem=1):
    import pytesseract
    config = f"--oem {oem} --psm {psm}"
    if allow:
        allow_sanitized = allow.replace(" ", "")
        # quote the -c value so spaces (if any) don't break parsing
        config += f" -c \"tessedit_char_whitelist={allow_sanitized}\""
    return pytesseract.image_to_string(img_bin, config=config, lang=lang).strip()

def _prep_text(bgr, invert=False):
    """Robust binarization for faint pencil on card stock."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # gentle contrast stretch
    lo, hi = np.percentile(g, (5, 95))
    if hi > lo:
        g = np.clip((g - lo) * (255.0/(hi - lo)), 0, 255).astype(np.uint8)
    g = cv2.medianBlur(g, 3)

    # try adaptive Gaussian; if very low variance, fall back to Otsu/Sauvola
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 9)
    # thicken strokes slightly
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)

    # If too sparse, fallback to Otsu
    if (thr == 255).mean() > 0.96 or (thr == 0).mean() > 0.96:
        _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)

    if invert: thr = 255 - thr
    # upsample small headers for Tesseract
    thr = _upsample_to_height(thr, target_h=70)
    return thr

def _read_best(img_bgr, psms, allow, oems=(1,)):  # LSTM only
    candidates = []
    for inv in (False, True):
        binimg = _prep_text(img_bgr, invert=inv)
        for psm in psms:
            for oem in oems:
                txt = _tess(binimg, psm=psm, allow=allow, oem=oem)
                score = len(re.findall(r"[A-Za-z0-9#$.\-]", txt)) - 0.5*len(re.findall(r"[_~^`]", txt))
                candidates.append((score, txt))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def _roi(img, x0, y0, x1, y1):
    """Crop by fractional coordinates (0..1)."""
    h, w = img.shape[:2]
    X0, Y0 = int(x0*w), int(y0*h)
    X1, Y1 = int(x1*w), int(y1*h)
    return img[Y0:Y1, X0:X1].copy()

def _upsample_to_height(img_bin, target_h=64):
    h, w = img_bin.shape[:2]
    if h >= target_h: return img_bin
    scale = target_h / float(h)
    return cv2.resize(img_bin, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def ocr_from_microcrops(micro: dict):
    """
    micro: {"left": Path, "scott": Path, "cat": Path, "sell": Path}
    Returns a dict compatible with CSV/JSON rows.
    """
    # read each crop (already grayscale on disk)
    left  = cv2.imread(str(micro["left"]),  cv2.IMREAD_GRAYSCALE)
    scott = cv2.imread(str(micro["scott"]), cv2.IMREAD_GRAYSCALE)
    cat   = cv2.imread(str(micro["cat"]),   cv2.IMREAD_GRAYSCALE)
    sell  = cv2.imread(str(micro["sell"]),  cv2.IMREAD_GRAYSCALE)

    # reuse your binarizer/upscaler and _tess wrapper
    def _best_txt(gray, psms, allow):
        # adapt to _prep_text signature (expects BGR), so fake a BGR for reuse
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return _read_best(bgr, psms=psms, allow=allow)

    cond_txt  = _best_txt(left,  psms=(8,7), allow="MNVLHU usedUSED ")
    year_txt  = _best_txt(left,  psms=(8,7), allow="0123456789")  # year is also on left
    scott_raw = _best_txt(scott, psms=(7,8), allow="#-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
    cat_raw   = _best_txt(cat,   psms=(7,8), allow="$0123456789., ")
    sell_raw  = _best_txt(sell,  psms=(7,8), allow="$0123456789., ")

    # --- Parse/normalize (same logic as ocr_card_structured) ---
    up = cond_txt.upper().replace(" ", "")
    cond_norm = None
    for tok in ("MNH", "MVLH", "MLH"):
        if tok in up:
            cond_norm = tok; break
    if cond_norm is None:
        cond_norm = "Used" if up == "" else None

    y = _clean_ascii(year_txt)
    m_year = re.search(r"\b(18|19|20)\d{2}\b", y)
    year = m_year.group(0) if m_year else None

    s_clean = _normalize_digits(_clean_ascii(scott_raw)).upper()
    m_scott = re.search(r"(?:#\s*)?([A-Z]{0,3}\d{1,4}(?:[A-Z]|-(?:[A-Z]?\d{1,4}))?)", s_clean)
    scott = m_scott.group(1) if m_scott else None

    cat_price  = _extract_amount(cat_raw)
    sell_price = _extract_amount(sell_raw)

    return {
        "condition_raw": cond_txt, "condition": cond_norm,
        "year_raw": year_txt,      "year": year,
        "scott_raw": scott_raw,    "scott": scott,
        "prices_raw": f"{cat_raw} | {sell_raw}",
        "cat_price": cat_price,    "sell_price": sell_price,
    }

def ocr_card_structured(img_path: pathlib.Path, debug=False, out_dir=None):
    bgr = cv2.imread(str(img_path))
    h, w = bgr.shape[:2]

    # ROIs (tuned to your cards)
    ROIS = {
        "condition": (0.02, 0.01, 0.25, 0.11),
        "year":      (0.02, 0.10, 0.25, 0.20),
        "scott":     (0.24, 0.02, 0.68, 0.14),   # a bit wider for #B100-B102
        "prices":    (0.62, 0.02, 0.98, 0.16),
    }
    crops = {k: _roi(bgr, *ROIS[k]) for k in ROIS}

    # --- OCR with field-specific configs ---
    cond_txt  = _read_best(crops["condition"], psms=(8,7), allow="MNVLHU usedUSED ")
    year_txt  = _read_best(crops["year"],      psms=(8,7), allow="0123456789")
    scott_raw = _read_best(crops["scott"],     psms=(7,8), allow="#-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

    # prices: split into Cat. (left) and Selling (right)
    pr = crops["prices"]
    midx = pr.shape[1] // 2
    left, right = pr[:, :midx], pr[:, midx:]
    cat_raw  = _read_best(left,  psms=(7,8), allow="$0123456789., ")
    sell_raw = _read_best(right, psms=(7,8), allow="$0123456789., ")

    # --- Parse/normalize ---
    # condition
    up = cond_txt.upper().replace(" ", "")
    cond_norm = None
    for tok in ("MNH", "MVLH", "MLH"):
        if tok in up:
            cond_norm = tok; break
    if cond_norm is None:
        cond_norm = "Used" if up == "" else None

    # year
    y = _clean_ascii(year_txt)
    m = re.search(r"\b(18|19|20)\d{2}\b", y)
    year = m.group(0) if m else None

    # Scott: accept #B100-B102, B100-102, MH239, A12a
    s_clean = _normalize_digits(_clean_ascii(scott_raw)).upper()
    # common fix: BIOO -> B100 etc handled by _normalize_digits
    m = re.search(r"(?:#\s*)?([A-Z]{0,3}\d{1,4}(?:[A-Z]|-(?:[A-Z]?\d{1,4}))?)", s_clean)
    scott = m.group(1) if m else None

    # prices
    cat_price  = _extract_amount(cat_raw)
    sell_price = _extract_amount(sell_raw)

    if debug and out_dir:
        vis = bgr.copy()
        for k,(x0,y0,x1,y1) in ROIS.items():
            X0,Y0,X1,Y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
            cv2.rectangle(vis, (X0,Y0), (X1,Y1), (0,255,0), 3)
            cv2.putText(vis, k, (X0, max(0,Y0-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imwrite(str(pathlib.Path(out_dir)/f"{img_path.stem}_roi_debug.jpg"), vis)

    return {
        "condition_raw": cond_txt, "condition": cond_norm,
        "year_raw": year_txt,      "year": year,
        "scott_raw": scott_raw,    "scott": scott,
        "prices_raw": f"{cat_raw} | {sell_raw}",
        "cat_price": cat_price,    "sell_price": sell_price,
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
def _crop_rel(parent_img, rel):
    ph, pw = parent_img.shape[:2]
    x0,y0,x1,y1 = rel
    X0,Y0 = int(x0*pw), int(y0*ph)
    X1,Y1 = int(x1*pw), int(y1*ph)
    return parent_img[Y0:Y1, X0:X1]

def _crop_frac(img_bgr, frac_box):
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = frac_box
    X0, Y0, X1, Y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
    X0, Y0 = max(0, X0), max(0, Y0)
    X1, Y1 = min(w, X1), min(h, Y1)
    return img_bgr[Y0:Y1, X0:X1].copy()

def header_microcrops(img_path: pathlib.Path, out_dir: pathlib.Path, jpeg_q=80):
    img = cv2.imread(str(img_path))  # this is the full header JPEG you saved
    paths = {}

    left  = cv2.cvtColor(_crop_rel(img, HEADER_LEFT),  cv2.COLOR_BGR2GRAY)
    scott = cv2.cvtColor(_crop_rel(img, HEADER_SCOTT), cv2.COLOR_BGR2GRAY)
    price = _crop_rel(img, HEADER_PRICE)

    cat   = cv2.cvtColor(_crop_rel(price, PRICE_CAT),  cv2.COLOR_BGR2GRAY)
    sell  = cv2.cvtColor(_crop_rel(price, PRICE_SELL), cv2.COLOR_BGR2GRAY)

    out_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in {"left":left, "scott":scott, "cat":cat, "sell":sell}.items():
        p = out_dir / f"{img_path.stem}_{key}.jpg"
        cv2.imwrite(str(p), arr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        paths[key] = p
    return paths

def make_header_crop(src_path: pathlib.Path,
                     dest_dir: pathlib.Path,
                     band=HEADER_BAND,
                     long_edge=1000,
                     jpeg_quality=78):
    """
    From a full dealer-card image, make a small grayscale JPEG of the header band.
    - Crops top band
    - Converts to grayscale
    - Resizes so the longer side is ~long_edge px
    - Saves compressed JPEG (no EXIF)
    Returns Path to the JPEG.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(src_path))
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {src_path}")
    crop = _crop_frac(bgr, band)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Resize keeping aspect ratio
    h, w = gray.shape[:2]
    scale = (long_edge / float(max(h, w))) if max(h, w) > long_edge else 1.0
    if scale != 1.0:
        gray = cv2.resize(gray, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_AREA)

    # Write grayscale JPEG (strip metadata by using OpenCV/Pillow re-encode)
    out_path = dest_dir / f"{src_path.stem}_HDR.jpg"
    # OpenCV writes grayscale JPEGs fine; quality 0..100
    cv2.imwrite(str(out_path), gray, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return out_path

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
      3. Make header crop (grayscale, compressed) for API use
      4. OCR and save results to CSV/JSON
    """
    scan_path = pathlib.Path(scan_path)
    out_dir = pathlib.Path(out_dir)
    raw_paths = detect_and_crop(str(scan_path), str(out_dir), rotate_mode)

    final_paths = []
    header_crops = []
    microcrops = []
    hdr_dir = out_dir / "AI_Crops"  # small, cheap payloads for API OCR

    for p in raw_paths:
        p = pathlib.Path(p)
        out_final = out_dir / f"{p.stem}_PS{p.suffix}"
        run_ps_action(p, out_final)
        final_paths.append(out_final)

        hdr = make_header_crop(out_final, hdr_dir,
                               band=HEADER_BAND,
                               long_edge=1000,
                               jpeg_quality=78)
        header_crops.append(hdr)

        micros_dir = hdr_dir / "micro"
        micros = header_microcrops(hdr, micros_dir)
        microcrops.append({"file": out_final.name, **micros})

    # OCR → build rows/json (no file I/O here)
    rows = []
    json_dump = {}
    for outp, micro in zip(final_paths, microcrops):
        if OCR_SOURCE == "ai":
            meta = ai_read_from_microcrops(
                {"left": micro["left"], "scott": micro["scott"], "cat": micro["cat"], "sell": micro["sell"]},
                model=AI_MODEL
            )
            # meta already matches SCHEMA; add *_raw placeholders for consistency
            meta = {
                "condition_raw": "", "year_raw": "", "scott_raw": "", "prices_raw": "",
                "condition": meta.get("condition") or None,
                "year": meta.get("year") or None,
                "scott": meta.get("scott") or None,
                "cat_price": meta.get("cat_price") or None,
                "sell_price": meta.get("sell_price") or None,
            }
        elif OCR_SOURCE == "micro":
            meta = ocr_from_microcrops(micro)
        else:  # "full"
            meta = ocr_card_structured(outp, debug=False, out_dir=str(out_dir))

        rows.append({
            "file": outp.name,
            "condition": meta.get("condition"),
            "year": meta.get("year"),
            "scott": meta.get("scott"),
            "cat_price": meta.get("cat_price"),
            "sell_price": meta.get("sell_price"),
        })
        json_dump[outp.name] = meta


    # You can also return header_crops so the next step can call the API on them.
    return final_paths, rows, json_dump  # header_crops available inside scope if you need it


# -----------------------------
# iter_images (make suffix check case-insensitive)
# -----------------------------
def iter_images(path: pathlib.Path, recursive: bool = False):
    """Yield absolute paths to JPG/JPEG/TIFF/PNG under path (file or folder)."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    path = path.resolve()
    if path.is_file():
        if path.suffix.lower() in exts:
            yield path
        return
    globber = path.rglob if recursive else path.glob
    for p in sorted(globber("*")):
        if p.suffix.lower() in exts:
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
    ap.add_argument("-o", "--out", default="./Stamps/Slices", help="output folder")
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