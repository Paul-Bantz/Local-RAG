
## Python requirements

```bash
pip install -r requirements.txt
```

## Building and running

Project can be run with compose - `podman compose up` or be ran as standalone containers :

### Ollama agent :
For best performance, run the container on a gpu.

```bash
podman build -t ollama-llama32-3b ollama.dockerfile
podman run -d \
--name ollama \
--replace \
--restart=always \
--device nvidia.com/gpu=all \
--security-opt=label=disable \
-p 11434:11434 \
-v ollama:/root/.ollama \
--stop-signal=SIGKILL \
ollama-llama32-3b
```

### Rag App :

```bash
podman build -t local-rag -f Dockerfiles/local_rag.dockerfile .
podman run -d \
--name local-rag \
--replace \
--restart=always \
-e TAVILY_API_KEY='<Tavily_API_Key_goes_here>' \
-e LLM_HOST='<LLM_host_url_goes_here>' \
-e LLM_MODEL='<Model_name_goes_here>' \
local-rag
```

