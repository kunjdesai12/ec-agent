import os
from llama_cpp import Llama
from app.config import QWEN_MODEL_PATH, ENABLE_TRANSLATION

translator_model = None

if ENABLE_TRANSLATION:
    print(f" Loading translation model: {QWEN_MODEL_PATH}")
    translator_model = Llama(
        model_path=str(QWEN_MODEL_PATH),
        n_ctx=2048,
        n_threads=max(os.cpu_count() // 2, 1),
        verbose=False,
    )
    print(" Translation model ready")


def translate_to_english(text: str) -> str:
    if not text or not text.strip() or not ENABLE_TRANSLATION or translator_model is None:
        return text

    system_prompt = (
    "You are a food-domain language normalizer. "
    "Convert the user's input — which may be in Hindi, Gujarati, "
    "Hinglish, or noisy spoken text — into clean, natural English. "
    "Preserve all food names, restaurant names, and product names exactly."
    )

    prompt = f"<|system|>\n{system_prompt}\n\n<|user|>\n{text}\n\n<|assistant|>\n"

    output = translator_model(prompt, max_tokens=128, temperature=0.5, top_p=0.9,
                              stop=["<|user|>", "<|system|>"])
    return output["choices"][0]["text"].strip()