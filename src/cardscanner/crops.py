import cv2, pathlib
from .config import HEADER_BAND, HEADER_PRICE, PRICE_CAT, PRICE_SELL, HEADER_LEFT, HEADER_SCOTT

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