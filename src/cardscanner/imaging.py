import pathlib, cv2

def rotate_90(img_bgr, mode="cw"):
    """Rotate an image 90° clockwise or counter-clockwise."""
    if mode == "cw":
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    else:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    

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