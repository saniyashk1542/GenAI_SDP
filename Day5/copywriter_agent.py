"""Copywriter agent: creates ad copy from the style guide."""

from __future__ import annotations

from pathlib import Path

from ollama_client import generate_text


def create_ad_copy(
    style_guide_path: str = "outputs/style_guide.txt",
    model: str = "llama3.2",
    output_path: str = "outputs/ad_copy.txt",
) -> str:
    """Read style guide, generate ad copy, and save it."""
    style_guide = Path(style_guide_path).read_text(encoding="utf-8")

    prompt = f"""
You are the Copywriter Agent in a creative agency.
Role: Write consistent ad copy based on the Director's style guide.

Style Guide:
{style_guide}

Task:
- Write:
  1) One headline
  2) One subheadline
  3) Three short social captions
  4) One call-to-action line

Rules:
- Match the tone words exactly from the style guide.
- Keep language clear and audience-focused.
- Do not invent a new tone.
"""

    ad_copy = generate_text(prompt=prompt, model=model, temperature=0.7)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ad_copy, encoding="utf-8")
    return ad_copy
