from ollama import Client
from pathlib import Path
from io import BytesIO
from PIL import Image

GARBAGE_PROMT = Path(__file__).with_name("garbage_filter_promt.txt").read_text(encoding="utf-8")
GLOABAL_IMAGE_PROMT = Path(__file__).with_name("extraction_prompt.txt").read_text(encoding="utf-8")

client = Client(host="http://127.0.0.1:11435")

def test():
    prompt = "Кратко объясни, что такое КПД солнечного элемента."

    result = client.generate(
        model="qwen3.5:9b",
        prompt=prompt,
        think=False,
        stream=False,
    )

    print(result.response)

def is_garbage(image_bytes):
    if is_too_small(image_bytes):
        return True

    image_for_model = image_bytes
    with Image.open(BytesIO(image_bytes)) as image:
        if max(image.size) > 768:
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            with BytesIO() as buffer:
                image.convert("RGB").save(buffer, format="PNG")
                image_for_model = buffer.getvalue()

    result = client.generate(
        model="qwen3.5:9b",
        prompt=GARBAGE_PROMT,
        images=[image_for_model],
        think=False,
        stream=False,
        options={"num_ctx": 2048, "num_predict": 8, "temperature": 0},
    )
    print(result.response)
    return result.response.strip().lower() == "garbage"


def is_too_small(image_bytes, min_width=128, min_height=128):
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
    return width < min_width or height < min_height