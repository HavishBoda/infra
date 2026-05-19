# Dispatch — Batched LLM Inference Server

Infergo is a production-inspired LLM inference server built in Go with a Python sidecar. It implements dynamic request batching, SSE streaming, Prometheus observability, and a thread-safe request queue.

---

## Overview

The system is composed of two services. The **Python sidecar** owns the model — it loads TinyLlama-1.1B via HuggingFace Transformers and exposes HTTP endpoints for single completions, batched completions, and streaming. It has no awareness of concurrency or multiple clients; it simply runs inference when asked.

The **Go server** owns everything else. It accepts client requests, manages a thread-safe queue, runs a dynamic batch scheduler, proxies SSE streams, and exposes Prometheus metrics. All concurrency logic, batching decisions, and observability live here. The two services communicate over HTTP, which means the sidecar can be swapped for a different model or inference backend without touching the Go layer.

---

## How to Run

### With Docker (recommended)

```bash
docker compose up --build
```

Both services start automatically. The Go server waits for the sidecar to be ready before accepting traffic.

### Manually

**Terminal 1 — Python sidecar:**
```bash
cd sidecar
pip install -r requirements.txt
python main.py
```

**Terminal 2 — Go server:**
```bash
cd goserver
go run main.go
```

### Verifying the Setup

```bash
# Single completion
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

# Streaming
curl -X POST http://localhost:8080/v1/completions/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

# Prometheus metrics
curl http://localhost:8080/metrics | grep inference
```

---

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/completions` | POST | Batched completion. Requests are queued and scheduled into batches before being forwarded to the sidecar. |
| `/v1/completions/stream` | POST | SSE streaming. Tokens are sent to the client as they are generated. Bypasses the queue entirely. |
| `/metrics` | GET | Prometheus metrics — queue depth, end-to-end latency histogram, and batch size distribution. |

---

## Design Decisions & Tradeoffs

### Separation of Concerns: Go Orchestration and Python Inference

The ML ecosystem is Python-native. HuggingFace Transformers, PyTorch, and the model weights themselves are all deeply tied to Python. Rather than working against this, the architecture separates concerns clearly: Python owns the model, Go owns orchestration.

This boundary has a practical benefit beyond language fit. The sidecar can be replaced independently — swapping TinyLlama for Phi-2, Mistral, or a quantized GGUF model requires no changes to the Go layer. In production, this is how companies like Replicate and Modal operate: a thin, language-agnostic orchestration layer in front of model workers that can be scaled and swapped independently.

### Concurrency Safety via a Single Worker Goroutine

The worker goroutine is the only execution context that ever calls the sidecar. This design choice provides concurrency safety without mutexes or shared memory. Any number of HTTP handlers can enqueue requests simultaneously, but the sidecar sees requests only through the single worker — one at a time, or one batch at a time.

The mechanism that makes this work is the `ResultChan` field on each `Request` struct. Every HTTP handler creates its own result channel, enqueues the request, and blocks waiting for a response. The worker sends the result back on the appropriate channel and the handler immediately unblocks. There is no polling, no shared state, and no coordination required between handlers.

```go
type Request struct {
    Prompt     string
    MaxTokens  int
    ResultChan chan Result
}
```

### Dynamic Batching

Neural networks are fundamentally matrix multiplications. A forward pass with eight prompts stacked into a matrix costs roughly the same compute as a forward pass with a single prompt, because the GPU or CPU performs the same underlying operations in parallel regardless of batch width. This makes batching one of the highest-leverage optimizations in inference serving.

The scheduler collects incoming requests using two configurable parameters:

```go
const (
    maxBatchSize = 8          // dispatch when the batch reaches this size
    maxWaitTime  = 20ms       // dispatch after this duration regardless of batch size
)
```

After the first request arrives, the scheduler opens a collection window. It continues accepting requests until either the batch is full or the timer expires, then dispatches the entire batch to the sidecar in a single call. This is implemented using Go's `select` statement on a channel and a timer, which handles the "wait up to N milliseconds or until a condition is met" logic cleanly and without busy-waiting.

As an additional optimization, if only one request arrives before the timeout, the scheduler calls the sidecar's `/complete` endpoint directly rather than `/batch_complete`, avoiding any batching overhead for single requests.

The core tradeoff is latency versus throughput. A larger `maxWaitTime` allows more requests to accumulate, improving throughput at the cost of individual request latency. A smaller value reduces latency but yields smaller batches and lower hardware utilization. These parameters can be tuned to match the latency and throughput requirements of a given deployment.

### Handling Variable max_tokens Across a Batch

Requests within the same batch may specify different `max_tokens` values. HuggingFace does not support per-prompt token limits natively for batched generation, so the scheduler takes the maximum `max_tokens` across all requests in the batch. Every prompt receives at least as many tokens as it requested; some receive slightly more.

A more sophisticated approach would be sequence-length-aware scheduling — grouping requests by similar prompt lengths and token budgets to minimize wasted compute from padding. This is a known limitation of the current implementation and a natural area for future improvement.

### Streaming and the Tension with Batching

Batching and streaming are architecturally in tension. Batching requires holding requests until a collection window closes before sending anything to the sidecar. Streaming requires forwarding tokens to the client the moment they are generated. These two requirements cannot be trivially satisfied simultaneously.

The architecture resolves this by separating the two into distinct modes. Requests to `/v1/completions` go through the queue and batch scheduler and receive a complete response when inference finishes. Requests to `/v1/completions/stream` bypass the queue entirely, going directly to the sidecar's `/stream_complete` endpoint, which uses HuggingFace's `TextIteratorStreamer` to yield tokens incrementally via a background thread. The Go handler reads the SSE stream from the sidecar and forwards each token to the client immediately, flushing after every write.

The deeper solution to combining streaming with batching is continuous batching — treating each token generation step as a unit of work rather than each request, so that new requests can be admitted mid-generation and completed requests can be retired without waiting for the rest of the batch. This is the approach taken by vLLM and is what makes it the dominant production inference backend. It requires a substantially more complex scheduler and is outside the scope of this implementation.

### Concurrency Limits of the Streaming Endpoint

Go handles concurrent streaming requests cleanly — each incoming request is served by its own goroutine with no shared state. However, concurrent streaming requests serialize at the Python sidecar because of the GIL. Under high concurrency on the streaming endpoint, each client experiences increased latency as requests queue inside the sidecar rather than in Go's managed queue.

In production this is addressed by running multiple sidecar instances behind a load balancer, or by using Python's async/await model with an ASGI server. For this implementation it is a known limitation.

### Per-Prompt Error Handling

When a batch request to the sidecar fails, all requests in that batch receive the same error. This is acceptable because the most common failure mode — the sidecar being unavailable — affects all prompts equally. A production system would implement per-prompt error handling, allowing partial results to be returned for prompts that succeeded. HuggingFace does not expose this natively for batched generation, so it would require splitting the batch on failure and retrying individual prompts.

### Docker and GPU Support

The Docker setup installs the CPU build of PyTorch to ensure portability across machines. This means containers always perform CPU inference even on hosts with available GPUs.

Enabling GPU support would require the CUDA build of PyTorch in the sidecar image, the NVIDIA Container Toolkit installed on the host, and `device_map="auto"` passed to the HuggingFace pipeline. On GPU hardware, the batching architecture becomes significantly more valuable — GPU parallelism means a batch of eight prompts completes in nearly the same wall-clock time as a single prompt, yielding throughput improvements of four to eight times at batch size eight.

---

## Benchmark Results

The benchmark script fires N concurrent requests against `/v1/completions` and measures end-to-end latency and throughput. All results were collected on CPU (Apple M3) with TinyLlama-1.1B and `max_tokens=32`.

| Concurrent Requests | p50 Latency | p95 Latency | p99 Latency | Throughput |
|---|---|---|---|---|
| 5 | 24.05s | 24.05s | 24.05s | 0.21 req/s |
| 10 | 39.00s | 49.26s | 49.26s | 0.20 req/s |
| 20 | 77.88s | 99.25s | 99.25s | 0.20 req/s |

**Throughput ceiling.** Throughput remains flat at approximately 0.20 req/s regardless of concurrent load. The bottleneck is CPU inference speed, not the Go server or the batch scheduler. The queue absorbs all concurrent requests without dropping any, and the scheduler continues operating correctly under load.

**Linear latency scaling.** As more requests accumulate in the queue, each request waits proportionally longer. p50 latency roughly doubles as concurrent requests double, which is consistent with a well-behaved queue draining at a fixed rate.

**Growing tail latency under load.** At five concurrent requests, p50 and p99 are identical because all requests fit into a single batch and complete together. At twenty concurrent requests, the gap between p50 and p99 grows to approximately twenty-one seconds, reflecting requests that must wait for earlier batches to complete before being dispatched.

**A note on sample size.** p95 and p99 converge at low sample counts. Statistically meaningful tail latency measurements require at least one thousand samples. These results are intended to demonstrate the system's behavior under load rather than serve as production-grade benchmarks.

**Expected GPU performance.** On GPU hardware, the same architecture would yield substantially higher throughput. Batching becomes the dominant optimization because GPU cores that sit idle during single-request inference are fully utilized across a batch. Expected throughput improvement at batch size eight: four to eight times the CPU baseline.

---

## Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `inference_queue_depth` | Gauge | Number of requests currently waiting in the queue |
| `inference_request_latency_seconds` | Histogram | End-to-end request latency, compatible with p50/p95/p99 queries |
| `inference_batch_size` | Histogram | Distribution of batch sizes dispatched to the sidecar |

Metrics are exposed at `/metrics` in the standard Prometheus text format and can be scraped by any Prometheus server and visualized in Grafana.

---

## Stack

- **Go** — HTTP server, request queue, dynamic batch scheduler, SSE proxy, Prometheus instrumentation
- **Python** — FastAPI sidecar, HuggingFace Transformers, TinyLlama-1.1B-Chat
- **Prometheus** — metrics collection and observability
- **Docker / Docker Compose** — containerization and service orchestration

---

## Known Limitations & Future Work

- **Continuous batching** — admit and retire requests mid-generation for higher GPU utilization, as implemented in vLLM
- **Sequence-length-aware scheduling** — group requests by similar lengths to minimize padding waste and improve batch efficiency
- **Multiple sidecar instances** — load balance streaming requests across workers to overcome the GIL bottleneck
- **Per-prompt error handling** — return partial batch results when individual prompts fail
- **GPU support** — CUDA PyTorch build with NVIDIA Container Toolkit integration
- **Authentication** — API key middleware on the Go server
