import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import MODEL_NAME
from prompts import SYSTEM_PROMPT, build_user_prompt
from schemas import build_schema


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/png"

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def extract_page_tables(client: OpenAI, image_path: Path) -> dict[str, Any]:
    data_url = image_to_data_url(image_path)
    schema = build_schema()

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_user_prompt(image_path.name)},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": True,
            }
        },
    )

    return json.loads(response.output_text)