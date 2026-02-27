"""Director agent: creates and saves a brand style guide."""

from __future__ import annotations

from pathlib import Path

from ollama_client import generate_text


def create_style_guide(
    brand: str,
    product: str,
    audience: str,
    model: str = "llama3.2",
    output_path: str = "outputs/style_guide.txt",
) -> str:
    """Generate a style guide and save it to a file."""
    prompt = f"""
You are the Director Agent in a creative agency.
Role: Build a concise brand style guide that all other agents must follow.

Brand: {brand}
Product: {product}
Audience: {audience}

Create a style guide using this exact structure:
1) Brand Summary (2-3 sentences)
2) Tone Words (exactly 5 words, comma-separated)
3) Voice Rules (4 bullet points)
4) Messaging Pillars (3 bullet points)
5) Do/Don't (2 Do bullets and 2 Don't bullets)

Important:
- Keep it practical and specific.
- Make Tone Words clear because all downstream agents must use them.
"""

    style_guide = generate_text(prompt=prompt, model=model, temperature=0.5)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(style_guide, encoding="utf-8")
    return style_guide
