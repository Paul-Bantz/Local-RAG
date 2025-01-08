
## Python requirements

```bash
pip install --quiet -U langchain langchain_community tiktoken langchain-nomic "nomic[local]" langchain-ollama scikit-learn langgraph tavily-python bs4 IPython
```

## Building and running

Running with compose :

```bash
podman compose up
```

Running as standalone containers : 

Ollama agent : 

```bash
podman build -t ollama-llama32-3b:latest ollama.dockerfile
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