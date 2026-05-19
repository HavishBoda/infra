import requests
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def send_request(prompt):
    start = time.time()
    requests.post("http://localhost:8080/v1/completions", json={"prompt": prompt, "max_tokens": 32})
    return time.time() - start

def benchmarker(n_requests):
    prompts = [
        "The capital of France is",
        "The largest planet is",
        "The speed of light is",
        "The first president of the United States was",
        "The chemical formula for water is",
    ]
    # repeat to fill n_requests
    sample_prompts = [prompts[i % len(prompts)] for i in range(n_requests)]

    # calculate throughout time
    start = time.time()
    with ThreadPoolExecutor(max_workers=n_requests) as executor:
        results = executor.map(send_request, sample_prompts)
        latencies = list(results)
        total_time = time.time() - start
        print(f"p50: {np.percentile(latencies, 50):.2f}s")
        print(f"p95: {np.percentile(latencies, 95):.2f}s")
        print(f"p99: {np.percentile(latencies, 99):.2f}s")
        print(f"throughput: {n_requests / total_time:.2f} req/s")

if __name__ == "__main__":
    print("--- 5 concurrent requests ---")
    benchmarker(5)
    print("\n--- 10 concurrent requests ---")
    benchmarker(10)
    print("\n--- 20 concurrent requests ---")
    benchmarker(20)