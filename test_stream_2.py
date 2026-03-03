from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')

# Simulating engine.py _stream_token behavior
tokens = tokenizer.encode('give it some space ', add_special_tokens=False)

prefix_len = 0
output_ids = []

for token in tokens:
    output_ids.append(token)
    
    # decode the current sequence, matching the engine behavior
    current_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    if len(current_text) > prefix_len:
        print(f"Yielding: `{current_text[prefix_len:]}` (Full text: `{current_text}`)")
        prefix_len = len(current_text)
