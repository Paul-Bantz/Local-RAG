# https://hub.docker.com/r/ollama/ollama
# podman build -t llama32-3b:latest -f ollama.dockerfile
FROM ollama/ollama

COPY ./scripts/ollama_entrypoint.sh /ollama_entrypoint.sh
RUN chmod +x /ollama_entrypoint.sh

EXPOSE 11434
ENTRYPOINT ["/bin/sh", "/ollama_entrypoint.sh"]