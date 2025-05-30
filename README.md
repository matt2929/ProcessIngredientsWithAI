build and run docker container

```shell

```

```
docker network create ollama-net 
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker build . -t test_ai_2
docker run --gpus all -it test_ai_2
docker run --network ollama-net -it test_ai_2
```