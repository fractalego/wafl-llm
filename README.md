# WAFL-llm

WAFL is built to run as a two-part system.
Both can be installed on the same machine.
This is the LLM side of the [WAFL project](https://github.com/fractalego/wafl).

![The two parts of WAFL](images/two-parts.png)


## LLM side (needs a GPU)

This is a model server for the speech-to-text model, the LLM, the embedding system, and the text-to-speech model.

#### Installation
In order to quickly run the LLM side, you can use the following installation commands:
```bash
pip install wafl-llm
wafl-llm start
```
which will use the default models and start the server on port 8080.

The installation will require MPI and Java installed on the system.
One can install both with the following commands
```bash
sudo apt install libopenmpi-dev
sudo apt install default-jdk
```

#### Configuration
A use-case specific configuration can be set by creating a `config.json` file in the path where `wafl-llm start` is executed.
The file should look like this (the default configuration)
```json
{
  "llm_model": "Deci/DeciLM-7B",
  "speaker_model": "facebook/fastspeech2-en-ljspeech",
  "whisper_model": "fractalego/personal-whisper-distilled-model",
  "sentence_embedder_models": "TaylorAI/gte-tiny"
}
```

The models are downloaded from the HugggingFace repository. Any other compatible model should work.


#### Docker
A docker image can be used to run it as in the following:

```bash
$ docker run -p8080:8080 --env NVIDIA_DISABLE_REQUIRE=1 --gpus all fractalego/wafl-llm:latest
```

or one can clone this repository and run the following

```bash
docker/build.sh
docker run -p8080:8080 --env NVIDIA_DISABLE_REQUIRE=1 --gpus all wafl-llm
```

