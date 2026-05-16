package main
import (
	"net/http"
	"encoding/json"
	"bytes"
	"log"
	"fmt"
	"time"
	"bufio"
)

const (
    maxBatchSize = 8
    maxWaitTime  = 20 * time.Millisecond
)

type CompletionRequest struct {
	Prompt string `json:"prompt"`
	MaxTokens int `json:"max_tokens"`
}

type BatchCompletionRequest struct {
    Prompts   []string `json:"prompts"`
    MaxTokens int      `json:"max_tokens"`
}

type Result struct {
	Completion string
	Err error
}

type Request struct {
    Prompt     string      `json:"prompt"`
    MaxTokens  int         `json:"max_tokens"`
    ResultChan chan Result
}

type SidecarResponse struct {
    Completion string `json:"completion"`
}

type IncomingRequest struct {
    Prompt    string `json:"prompt"`
    MaxTokens int    `json:"max_tokens"`
}

// requests sidecar client with the LLM prompt + max tokens
func SidecarClient(prompt string, maxTokens int) (string, error) {
	req := CompletionRequest{Prompt: prompt, MaxTokens: maxTokens}
	data, err := json.Marshal(req)
	if (err != nil){
		return "", err
	}

	resp, err := http.Post("http://localhost:8000/complete", "application/json", bytes.NewReader(data))
	if (err != nil){
		return "", err
	}

	var result SidecarResponse
	json.NewDecoder(resp.Body).Decode(&result)

	return result.Completion, nil
}

func BatchSidecarClient(requests []Request) ([]string, error) {
	prompts := []string{}
	maxTokens := 0

	for _, req := range requests {
		prompts = append(prompts, req.Prompt)
		maxTokens = max(maxTokens, req.MaxTokens)
	}

	// request to post to sidecar
	request := BatchCompletionRequest{Prompts: prompts, MaxTokens: maxTokens}
	data, err := json.Marshal(request)
	if (err != nil){
		return nil, err
	}

	resp, err := http.Post("http://localhost:8000/batch_complete", "application/json", bytes.NewReader(data))
	if (err != nil){
		return nil, err
	}

	var result []string
	json.NewDecoder(resp.Body).Decode(&result)

	return result, nil
}

// will run in own goroutine, checks for http requests
func Worker(queue chan Request) {
	for {
        // block until first request arrives
        first := <-queue
        batch := []Request{first}
        timer := time.NewTimer(maxWaitTime)

        // inner loop - collect more requests until timeout or full
        inner:
        for {
            select {
            case req := <-queue:
                // add req to batch
				batch = append(batch, req)

				// if batch is full, break to inner label
				if (len(batch) == maxBatchSize){
					break inner
				}
			// timer fires after "maxWaitTime" by sending on channel 'C'
            case <-timer.C:
                // break to inner label
				break inner
            }
        }

        // send the batch here
		if (len(batch) == 1){
			fmt.Println("Sending to sidecar, only 1 request\n")
			res, err := SidecarClient(batch[0].Prompt, batch[0].MaxTokens)
			if (err != nil){
				batch[0].ResultChan <- Result{Err: err}
			} else {
				batch[0].ResultChan <- Result{Completion: res}
			}
		} else {
			fmt.Printf("Processing batch of %d requests\n", len(batch))
			results, err := BatchSidecarClient(batch)
			if (err != nil){
				for _, req := range batch {
					req.ResultChan <- Result{Err: err}
				}
			} else {
				for i, completion := range results {
					batch[i].ResultChan <- Result{Completion: completion}
				}
			}
		}
    }
}

func main() {
	fmt.Printf("Starting server...")
    queue := make(chan Request, 100)
	// launch worker
	go Worker(queue)

    http.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
        var incoming IncomingRequest
		json.NewDecoder(r.Body).Decode(&incoming)
		resultChan := make(chan Result, 1)
		queue <- Request{Prompt: incoming.Prompt, MaxTokens: incoming.MaxTokens, ResultChan: resultChan}

		result := <-resultChan
		if result.Err != nil {
			http.Error(w, result.Err.Error(), http.StatusInternalServerError)
    		return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
    })

	http.HandleFunc("/v1/completions/stream", func(w http.ResponseWriter, r *http.Request) {
		var incoming IncomingRequest
		json.NewDecoder(r.Body).Decode(&incoming)

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// marshal incoming into JSON and POST 
		data, err := json.Marshal(incoming)
		if (err != nil){
			return
		}

		resp, err := http.Post("http://localhost:8000/stream_complete", "application/json", bytes.NewReader(data))
		if (err != nil){
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			fmt.Fprintf(w, "%s\n", line)
			flusher.Flush()
		}

	})

	// start server
	log.Fatal(http.ListenAndServe(":8080", nil))
}