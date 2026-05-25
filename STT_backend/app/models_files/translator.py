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
    "You are a strict language normalizer for a food delivery application. "
    "Your only job is to convert the user's input into clean, natural English. "
    "Do not add any information that is not present in the original input. "
    "Do not explain, summarize, or respond conversationally. "
    "Output only the translated English sentence and nothing else. "
    "All restaurant names and food item names in the input belong to a "
    "verified food delivery database. "
    "in the input — do not translate, paraphrase, or modify them in any way. "
    "If the input is already in English, return it as-is with only minor "
    "cleanup like removing filler words or fixing speech recognition errors. "
    "Never add greetings, explanations, or extra sentences to your output."
)

    prompt = (
    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    f"<|im_start|>user\n{text}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
    output = translator_model(
    prompt,
    max_tokens=64,
    temperature=0.0,
    top_p=1.0,
    stop=["<|im_end|>", "<|im_start|>"]
)
    return output["choices"][0]["text"].strip()