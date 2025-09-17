import re
from dotenv import load_dotenv
load_dotenv()

OCR_SOURCE = "micro"         # "micro" | "full" | "ai"
AI_MODEL   = "gpt-4.1"

PHOTOSHOP_BUNDLE_ID = "com.adobe.Photoshop"
ACTION_SET  = "Default Actions"
ACTION_NAME = "Calibrate Scanned Image"
JPEG_QUALITY = 10

USE_TESSERACT = True
TESS_LANGS = "eng"
ROTATE_MODE = "cw"
DEFAULT_ROTATE_MODE = "cw"
DPI_FOR_OCR = 300
BIN_THR_PCT = (5, 95)

HEADER_BAND  = (0.0, 0.0, 1.0, 0.2)
HEADER_LEFT  = (0.00, 0.00, 0.18, 0.8)
HEADER_SCOTT = (0.18, 0.00, 0.53, 0.8)
HEADER_PRICE = (0.52, 0.00, 1.00, 1.0)
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

LOOKALIKE_MAP = str.maketrans({"O":"0","o":"0","I":"1","l":"1","|":"1","S":"5","s":"5","Z":"2","z":"2"})
SCOTT_RX = re.compile(r"(?:#\s*)?([A-Z]{0,3}\d{1,4}(?:[A-Z]|-(?:[A-Z]?\d{1,4}))?)")
YEAR_RX = re.compile(r"\b(18|19|20)\d{2}\b")
