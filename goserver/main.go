package main
import (
	"net/http"
	"encoding/json"
	"bytes"
	"log"
	"fmt"
)

type CompletionRequest struct {
	Prompt string `json:"prompt"`
	MaxTokens int `json:"max_tokens"`
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

// requests sidecar client with the LLM prompt + max tokens
func SidecarClient(prompt string, maxTokens int) (string, error) {
	req := CompletionRequest{Prompt: prompt, MaxTokens: maxTokens}
	data, err := json.Marshal(req)
	fmt.Println("Sending to sidecar: ", string(data))
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

// will run in own goroutine, checks for http requests
func Worker(queue chan Request) {
	for req := range queue {
		res, err := SidecarClient(req.Prompt, req.MaxTokens)
		if (err != nil){
			req.ResultChan <- Result{Err: err}
		} else {
			req.ResultChan <- Result{Completion: res}
		}
	}
}

func main() {
	fmt.Printf("Starting server...")
    queue := make(chan Request, 100)
	// launch worker
	go Worker(queue)

    http.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
        var req Request
		json.NewDecoder(r.Body).Decode(&req)

		resultChan := make(chan Result, 1)

		queue <- Request{Prompt: req.Prompt, MaxTokens: req.MaxTokens, ResultChan: resultChan}

		result := <-resultChan

		json.NewEncoder(w).Encode(result)
    })

	// start server
	log.Fatal(http.ListenAndServe(":8080", nil))
}