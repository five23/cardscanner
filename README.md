# Dealer Card Scanner Pipeline

This is a Python pipeline for digitizing stamp dealer cards. It's still very experimental. Your mileage will vary.
It takes high-resolution scans of 2x2 (or larger) dealer card sheets, crops each card out, runs a Photoshop action for calibration, and applies OCR to extract structured data (condition, year, Scott number, catalog price, and selling price).

---

## Features

- **Card detection**: Uses OpenCV to find white dealer cards against a black background.
- **Perspective correction**: Crops and warps each card to a straight rectangle.
- **Photoshop integration**: Calls Adobe Photoshop via AppleScript + JSX to run a custom Action (e.g. color/white balance).
- **OCR with Tesseract**: Reads fields from the card (Condition, Year, Scott #, Catalog Price, Selling Price).
- **Structured output**: Produces both `cards_ocr.csv` and `cards_ocr.json`.
- **Debug overlays**: Optional saved images showing detected contours/ROIs for tuning.

---

## Requirements

- macOS with **Adobe Photoshop** installed  
- Python 3.9+  
- Tesseract OCR installed and available on PATH  

Python dependencies:

```txt
opencv-python
numpy
pillow
pytesseract
