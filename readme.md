
# Local RAG Demo

This project is an extension of [Local RAG agent with LLaMA3](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/)

This is a demo project showcasing the use of Langchain in a RAG context. It relies on an Ollama server running a Llama3 llm.

Building blocks :
- [streamlit](https://streamlit.io/) GUI for LLM interaction
- [In-memory](https://python.langchain.com/docs/integrations/vectorstores/sklearn/) vector store (see [Rag App](#rag-app))
- [Nomic Embedding model](https://www.nomic.ai/blog/posts/local-nomic-embed)
- Extra data used for answer correctness are provided through websearch using the [Tavily API](https://tavily.com/).

## Running locally

- The app requires a local ollama server running
- See .vscode/launch.json for running the RAG app locally. The environment variables to set are the same as [Environment variables](#environment-variables).
Entrypoint is `streamlit streamlit_app.py` (see [Running your streamlit app](https://docs.streamlit.io/develop/concepts/architecture/run-your-app))

## Running in containers

### Via compose

```bash
docker-compose -f .\Dockerfiles\compose.yaml -p "local-rag-stack" up
```

### Manually

#### Ollama agent :

Run the container on a dedicated network (`docker network create rag-network`)
```bash
docker build -t llama32-3b -f Dockerfiles/ollama.dockerfile Dockerfiles

docker run -d \
--name ollama \
--gpus=all \
--network=rag-network \
--restart=always \
-p 11434:11434 \
-v ollama:/root/.ollama \
--stop-signal=SIGKILL \
llama32-3b
```

#### RAG App :

```bash
docker build -t local-rag -f Dockerfiles/local_rag.dockerfile .

docker run -d \
--name local-rag \
--gpus=all \
--network=rag-network \
--restart=always \
--security-opt=label=disable \
--network="host" `# Necessary if the ollama server is running locally ie at the hosts localhost` \
-p 8501:8501 \
-e GPU_DEVICE='cuda' \
-e NOMIC_EMBEDDING_MODEL='nomic-embed-text-v1.5' \
-e TAVILY_API_KEY='<Tavily_API_Key_goes_here>' \
-e LLM_HOST='<LLM_host_url_goes_here>' \
-e LLM_MODEL='<Model_name_goes_here>' \
local-rag
```

The server runs at [http://localhost:8501](http://localhost:8501)

#### Environment variables

- **GPU_DEVICE** : target device to run the nomic embedding model on
- **NOMIC_EMBEDDING_MODEL** : the nomic embedding model version (see [Huggingface](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5))
- **TAVILY_API_KEY** : your [Tavily](https://pages.github.com/) API Key for the websearch functionnality
- **LLM_HOST** : host and port of the server running your model ([http://localhost:11434](http://localhost:11434 by default)
- **LLM_MODEL** : name of the model running on the ollama instance (**llama3.2:3b-instruct-fp16** if using the provided ollama.dockerfile)
