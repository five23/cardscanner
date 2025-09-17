import cv2, pathlib, numpy as np, re
from . import __init__
from ..config import TESS_LANGS, BIN_THR_PCT
from ..parsing import clean_ascii, normalize_digits, extract_amount, parse_scott, parse_year, normalize_condition, apply_handwriting_rules

def _tess(img_bin, psm=6, allow=None, lang=TESS_LANGS, oem=1):
    import pytesseract
    config = f"--oem {oem} --psm {psm}"
    if allow:
        allow = allow.replace(" ", "")
        config += f' -c "tessedit_char_whitelist={allow}"'
    return pytesseract.image_to_string(img_bin, config=config, lang=lang).strip()

def _prep_text(bgr, invert=False):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(g, BIN_THR_PCT)
    if hi > lo:
        g = np.clip((g - lo) * (255.0/(hi - lo)), 0, 255).astype(np.uint8)
    g = cv2.medianBlur(g, 3)
    thr = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    if (thr==255).mean()>0.96 or (thr==0).mean()>0.96:
        _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if invert: thr = 255 - thr
    h,w = thr.shape[:2]
    if h < 64:
        scale = 64/float(h)
        thr = cv2.resize(thr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return thr

def _read_best(img_bgr, psms, allow, oems=(1,)):
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

def _roi(img, x0,y0,x1,y1):
    h,w = img.shape[:2]
    X0,Y0,X1,Y1 = int(x0*w),int(y0*h),int(x1*w),int(y1*h)
    return img[Y0:Y1, X0:X1].copy()

def ocr_from_microcrops(micro: dict):
    left  = cv2.imread(str(micro["left"]),  cv2.IMREAD_GRAYSCALE)
    scott = cv2.imread(str(micro["scott"]), cv2.IMREAD_GRAYSCALE)
    cat   = cv2.imread(str(micro["cat"]),   cv2.IMREAD_GRAYSCALE)
    sell  = cv2.imread(str(micro["sell"]),  cv2.IMREAD_GRAYSCALE)

    left  = apply_handwriting_rules(left,  "left")
    scott = apply_handwriting_rules(scott, "scott")
    cat   = apply_handwriting_rules(cat,   "cat")
    sell  = apply_handwriting_rules(sell,  "sell")

    def _best_txt(gray, psms, allow):
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return _read_best(bgr, psms=psms, allow=allow)

    cond_txt  = _best_txt(left,  (8,7), "MNVLHU usedUSED ")
    year_txt  = _best_txt(left,  (8,7), "0123456789")
    scott_raw = _best_txt(scott, (7,8), "#-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
    cat_raw   = _best_txt(cat,   (7,8), "$0123456789., ")
    sell_raw  = _best_txt(sell,  (7,8), "$0123456789., ")

    return {
        "condition_raw": cond_txt, "condition": normalize_condition(cond_txt),
        "year_raw": year_txt,      "year": parse_year(year_txt),
        "scott_raw": scott_raw,    "scott": parse_scott(scott_raw),
        "prices_raw": f"{cat_raw} | {sell_raw}",
        "cat_price":  extract_amount(cat_raw),
        "sell_price": extract_amount(sell_raw),
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
    y = clean_ascii(year_txt)
    m = re.search(r"\b(18|19|20)\d{2}\b", y)
    year = m.group(0) if m else None

    # Scott: accept #B100-B102, B100-102, MH239, A12a
    s_clean = normalize_digits(clean_ascii(scott_raw)).upper()
    # common fix: BIOO -> B100 etc handled by _normalize_digits
    m = re.search(r"(?:#\s*)?([A-Z]{0,3}\d{1,4}(?:[A-Z]|-(?:[A-Z]?\d{1,4}))?)", s_clean)
    scott = m.group(1) if m else None

    # prices
    cat_price  = extract_amount(cat_raw)
    sell_price = extract_amount(sell_raw)

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
