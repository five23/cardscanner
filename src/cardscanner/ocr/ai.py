import json
from openai import OpenAI
from dotenv import load_dotenv
from ..parsing import img_part
from ..config import SCHEMA

load_dotenv()

client = OpenAI()

def ai_read_from_microcrops(micro: dict, model="gpt-4o-mini"):
    """
    micro: {"left": Path, "scott": Path, "cat": Path, "sell": Path}
    Returns dict matching SCHEMA.
    """
    imgs = [
        img_part(micro["left"]),
        img_part(micro["scott"]),
        img_part(micro["cat"]),
        img_part(micro["sell"]),
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
