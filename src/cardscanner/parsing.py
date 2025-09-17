import re, pathlib, base64
from .config import SCOTT_RX, YEAR_RX, LOOKALIKE_MAP

def clean_ascii(s: str) -> str:
    return re.sub(r"[^\w\-\s#$.,/]", "", s)

def normalize_digits(s: str) -> str:
    def fix(m): return m.group(0).translate(LOOKALIKE_MAP)
    return re.sub(r"([$]?\s*[A-Za-z]*\d[\dOIlSsz.,]*)", fix, s)

def extract_amount(txt: str):
    t = normalize_digits(clean_ascii(txt)).replace(" ", "").replace(",", ".")
    m = re.search(r"\$?\d+(?:\.\d{1,2})?", t)
    if not m: return None
    amt = m.group(0).lstrip("$")
    if "." not in amt: amt += ".00"
    return f"${amt}"

def parse_scott(s: str):
    s2 = normalize_digits(clean_ascii(s)).upper()
    m = SCOTT_RX.search(s2)
    return m.group(1) if m else None

def parse_year(s: str):
    m = YEAR_RX.search(clean_ascii(s))
    return m.group(0) if m else None

def normalize_condition(raw: str):
    up = (raw or "").upper().replace(" ", "")
    for tok in ("MNH","MVLH","MLH"):
        if tok in up: return tok
    return "Used" if up == "" else None

def apply_handwriting_rules(pre_ocr_gray_image, region_name: str):
    """
    Hook for rule-based tweaks BEFORE OCR.
    Examples:
      - If region_name=="scott", dilate slightly to connect '1' stems.
      - If region_name=="price", erode to thin blobs that look like '$S' => '$5'.
    Return possibly modified image.
    """
    # start simple — no-op by default
    return pre_ocr_gray_image

def img_part(path: pathlib.Path):
    # pick a sensible MIME (JPEG default)
    ext = path.suffix.lower()
    mime = "image/png" if ext in {".png"} else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}