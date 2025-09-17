import os, re, cv2, pathlib
import numpy as np

from PIL import Image
from .config import HEADER_BAND, OCR_SOURCE
from .imaging import rotate_90
from .photoshop import run_ps_action
from .crops import make_header_crop, header_microcrops
from .ocr.tesseract import ocr_from_microcrops, ocr_card_structured
from .ocr.ai import ai_read_from_microcrops

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

def process_scan(scan_path: str, out_dir: str, rotate_mode: str):
    scan_path = pathlib.Path(scan_path)
    out_dir = pathlib.Path(out_dir)
    raw_paths = detect_and_crop(str(scan_path), str(out_dir), rotate_mode)

    final_paths, microcrops = [], []
    hdr_dir = out_dir / "AI_Crops"

    for p in raw_paths:
        p = pathlib.Path(p)
        out_final = out_dir / f"{p.stem}_PS{p.suffix}"
        run_ps_action(p, out_final)
        final_paths.append(out_final)

        hdr = make_header_crop(out_final, hdr_dir, band=HEADER_BAND, long_edge=1000, jpeg_quality=78)
        micros = header_microcrops(hdr, hdr_dir / "micro")
        microcrops.append({"file": out_final.name, **micros})

    rows, json_dump = [], {}
    for outp, micro in zip(final_paths, microcrops):
        if OCR_SOURCE == "ai":
            meta_ai = ai_read_from_microcrops(
                {"left": micro["left"], "scott": micro["scott"], "cat": micro["cat"], "sell": micro["sell"]}
            )
            meta = {
                "condition_raw":"", "year_raw":"", "scott_raw":"", "prices_raw":"",
                "condition": meta_ai.get("condition") or None,
                "year":      meta_ai.get("year") or None,
                "scott":     meta_ai.get("scott") or None,
                "cat_price": meta_ai.get("cat_price") or None,
                "sell_price":meta_ai.get("sell_price") or None,
            }
        elif OCR_SOURCE == "micro":
            meta = ocr_from_microcrops(micro)
        else:
            meta = ocr_card_structured(outp, debug=False, out_dir=str(out_dir))

        rows.append({k: meta.get(k) for k in ("condition","year","scott","cat_price","sell_price")}|{"file": outp.name})
        json_dump[outp.name] = meta

    return final_paths, rows, json_dump
