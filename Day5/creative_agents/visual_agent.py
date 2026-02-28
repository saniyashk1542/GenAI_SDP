"""Visual agent: creates image prompts from the style guide."""

from __future__ import annotations

from pathlib import Path

from ollama_client import generate_text


def create_image_prompts(
    style_guide_path: str = "outputs/style_guide.txt",
    model: str = "llama3.2",
    output_path: str = "outputs/image_prompts.txt",
) -> str:
    """Read style guide, generate image prompts, and save them."""
    style_guide = Path(style_guide_path).read_text(encoding="utf-8")

    prompt = f"""
You are the Visual Agent in a creative agency.
Role: Create visual direction prompts for an image generation tool.

Style Guide:
{style_guide}

Task:
- Provide 4 image prompts.
- Each prompt should include:
  1) Main scene
  2) Subject details
  3) Lighting/color direction
  4) Composition/camera style

Rules:
- Use the exact tone words from the style guide.
- Keep prompts brand-consistent and ad-ready.
- Keep each prompt to 2-4 lines.
"""

    image_prompts = generate_text(prompt=prompt, model=model, temperature=0.7)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(image_prompts, encoding="utf-8")
    return image_prompts
