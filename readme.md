
# Local RAG Demo

This project is an extension of [Local RAG agent with LLaMA3](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/)

This is a demo project showcasing the use of Langchain in a RAG context. It relies on an Ollama server running a Llama3 llm.

Building blocks :
- [streamlit](https://streamlit.io/) GUI for LLM interaction
- [In-memory](https://python.langchain.com/docs/integrations/vectorstores/sklearn/) vector store (see [Rag App](#rag-app))
- [Nomic Embedding model](https://www.nomic.ai/blog/posts/local-nomic-embed)
- Extra data used for answer correctness are provided through websearch using the [Tavily API](https://tavily.com/).

## Building and running

### Ollama agent :
For best performance, run the container on a gpu.

```bash
podman build -t llama32-3b -f Dockerfiles/ollama.dockerfile Dockerfiles
podman run -d \
--name ollama \
--replace \
--restart=always \
--device nvidia.com/gpu=all \
--security-opt=label=disable \
-p 11434:11434 \
-v ollama:/root/.ollama \
--stop-signal=SIGKILL \
llama32-3b
```

### Rag App :

```bash
streamlit run LocalRag/streamlit_app.py
```

#### Environment variables

- **TAVILY_API_KEY** : your [Tavily](https://pages.github.com/) API Key for the websearch functionnality
- **LLM_HOST** : host and port of the server running your model ([http://localhost:11434](http://localhost:11434 by default)
- **LLM_MODEL** : name of the model running on the ollama instance (**llama3.2:3b-instruct-fp16** if using the provided ollama.dockerfile)
