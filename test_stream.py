from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
tokens = tokenizer.encode('give it some space ', add_special_tokens=False)

print("Tokens:", tokens)

prefix_len = 0
output_ids = []

for token in tokens:
    output_ids.append(token)
    current_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    if len(current_text) > prefix_len:
        print(f"Yielding ({token}): `{current_text[prefix_len:]}`")
        prefix_len = len(current_text)
