"""Run the Creative Agency Pipeline end-to-end."""

from __future__ import annotations

from pathlib import Path

from copywriter_agent import create_ad_copy
from director_agent import create_style_guide
from visual_agent import create_image_prompts


def extract_tone_words(style_guide: str) -> list[str]:
    """Extract tone words from the 'Tone Words' line in the style guide."""
    for line in style_guide.splitlines():
        lowered = line.lower()
        if "tone words" in lowered:
            # Supports lines like:
            # "2) Tone Words: bold, warm, clear, modern, trusted"
            after_colon = line.split(":", 1)
            if len(after_colon) < 2:
                continue
            words = [w.strip().lower() for w in after_colon[1].split(",")]
            words = [w for w in words if w]
            if words:
                return words
    return []


def outputs_exist() -> bool:
    required = [
        Path("outputs/style_guide.txt"),
        Path("outputs/ad_copy.txt"),
        Path("outputs/image_prompts.txt"),
    ]
    return all(path.exists() for path in required)


def tone_words_shared(tone_words: list[str], ad_copy: str, image_prompts: str) -> bool:
    if not tone_words:
        return False

    ad_lower = ad_copy.lower()
    visual_lower = image_prompts.lower()
    return all(word in ad_lower and word in visual_lower for word in tone_words)


def main() -> None:
    print("=== Creative Agency Pipeline ===")
    brand = input("Enter brand name: ").strip()
    product = input("Enter product/service: ").strip()
    audience = input("Enter target audience: ").strip()
    model = input("Enter Ollama model (press Enter for llama3.2): ").strip() or "llama3.2"

    style_guide = create_style_guide(
        brand=brand,
        product=product,
        audience=audience,
        model=model,
    )
    ad_copy = create_ad_copy(model=model)
    image_prompts = create_image_prompts(model=model)

    tone_words = extract_tone_words(style_guide)
    success = outputs_exist() and tone_words_shared(tone_words, ad_copy, image_prompts)

    print("\nPipeline finished.")
    print("Generated files:")
    print("- outputs/style_guide.txt")
    print("- outputs/ad_copy.txt")
    print("- outputs/image_prompts.txt")
    print(f"Extracted tone words: {tone_words if tone_words else 'Not found'}")
    print(f"Success check: {'PASS' if success else 'CHECK OUTPUTS'}")


if __name__ == "__main__":
    main()
