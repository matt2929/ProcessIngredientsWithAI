build and run docker container

![Screenshot 2025-06-09 at 11.49.23 PM.png](cluster_topic_modeling_emotions%2Fimages%2FScreenshot%202025-06-09%20at%2011.49.23%E2%80%AFPM.png)

```shell
 source env/bin/activate\n
 python --version
 cd cluster_topic_modeling_emotions
 pip install -r requirements.txt

```

```shell
docker network create ollama-net 
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker build . -t test_ai_2
docker run --gpus all -it test_ai_2
docker run --network ollama-net -it test_ai_2
```