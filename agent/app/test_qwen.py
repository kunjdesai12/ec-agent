from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
print(tok.apply_chat_template([
    {"role":"system","content":"BASE"},
    {"role":"user","content":"hi"},
    {"role":"system","content":"[ORDER STATUS] 1x Pav Bhaji ₹120"},
    {"role":"user","content":"add another"},
], tokenize=False, add_generation_prompt=True))