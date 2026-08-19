Run following command inside the current sub-directory (where `dockerfile` is located).

Build this docker container with `docker build -t llm-sandbox .`.

Start this dockerfile with `docker run -d --name llm-sandbox -v $(pwd):/workspace llm-sandbox` for it to be a persistent docker.

Run `python sandbox_runner_example.py` to test.

# if it is already running

```
docker stop llm-sandbox
docker rm llm-sandbox
docker run -d --name llm-sandbox -v $(pwd):/workspace llm-sandbox
```