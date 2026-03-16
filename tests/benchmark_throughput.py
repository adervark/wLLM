import argparse
import time
import torch
import logging
from winllm.engine import InferenceEngine, GenerationRequest
from winllm.config import ModelConfig, KVCacheConfig, SamplingParams, QuantizationType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_benchmark(model_path, use_compile, num_prompts, max_tokens):
    model_config = ModelConfig(
        model_name_or_path=model_path,
        quantization=QuantizationType.NF4,
        compile=use_compile,
        max_model_len=2048
    )
    kv_config = KVCacheConfig()
    
    engine = InferenceEngine(model_config, kv_config)
    print(f"\n[BENCHMARK] Loading model: {model_path} (compile={use_compile})...")
    engine.load_model()
    
    prompts = [
        "Explain quantum entanglement like I'm five.",
        "Write a 500-word essay on the impact of AI on modern society.",
        "Translate the following English text to French: 'Hello, how are you today? I hope you have a wonderful morning.'",
        "Summarize the plot of the movie Inception.",
        "Write a Python script to scrape a website using BeautifulSoup."
    ][:num_prompts]
    
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.7)
    
    # Warmup
    if use_compile:
        print("[BENCHMARK] Performing warmup pass for torch.compile...")
        warmup_req = GenerationRequest(prompt="Hello", sampling_params=SamplingParams(max_tokens=5))
        engine.generate(warmup_req)
        print("[BENCHMARK] Warmup complete.")

    total_gen_tokens = 0
    total_time = 0.0
    ttfts = []
    
    print(f"[BENCHMARK] Running {len(prompts)} prompts...")
    
    for i, p in enumerate(prompts):
        req = GenerationRequest(prompt=p, sampling_params=sampling)
        
        # We need a custom callback to measure TTFT
        start_time = time.perf_counter()
        first_token_time = None
        
        def on_token(text, finished):
            nonlocal first_token_time
            if not first_token_time and not finished:
                first_token_time = time.perf_counter()

        req._stream_callback = on_token
        
        result = engine.generate(req)
        end_time = time.perf_counter()
        
        gen_time = end_time - start_time
        total_time += gen_time
        total_gen_tokens += result.generation_tokens
        
        if first_token_time:
            ttft = (first_token_time - start_time) * 1000
            ttfts.append(ttft)
            ttft_str = f"{ttft:.2f}ms"
        else:
            ttft_str = "N/A"
            
        tps = result.generation_tokens / gen_time
        print(f"  Prompt {i+1}: {result.generation_tokens} tokens, TTFT: {ttft_str}, TPS: {tps:.2f}")

    avg_tps = total_gen_tokens / total_time
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    
    print(f"\n{'='*40}")
    print(f"RESULTS (compile={use_compile})")
    print(f"{'='*40}")
    print(f"Average TPS: {avg_tps:.2f}")
    print(f"Average TTFT: {avg_ttft:.2f}ms")
    print(f"Total Time: {total_time:.2f}s")
    print(f"{'='*40}\n")
    
    engine.unload_model()
    return avg_tps, avg_ttft

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    
    print("Starting Comparison Benchmark...")
    
    # Baseline
    tps_base, ttft_base = run_benchmark(args.model, False, args.num_prompts, args.max_tokens)
    
    # Optimized
    tps_opt, ttft_opt = run_benchmark(args.model, True, args.num_prompts, args.max_tokens)
    
    print("\nFINAL COMPARISON")
    print("-" * 40)
    print(f"TPS Improvement: {(tps_opt/tps_base - 1)*100:.1f}%")
    print(f"TTFT Reduction: {(1 - ttft_opt/ttft_base)*100:.1f}%")
    print("-" * 40)
