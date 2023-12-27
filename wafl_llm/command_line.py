import os
import sys

from wafl_llm.variables import get_variables

_path = os.path.dirname(__file__)


def print_incipit():
    print()
    print(f"Running WAFL_LLM version {get_variables()['version']}.")
    print()


def print_help():
    print("\n")
    print("These are the available commands:")
    print("> wafl_llm start: Initialize the current folder")
    print()


def add_cwd_to_syspath():
    sys.path.append(os.getcwd())


def start_llm_server():
    services = ["llm", "sentence_embedder", "whisper", "speaker"]
    if not os.path.exists(f"models"):
        os.system(f"mkdir -p models")

    for service in services:
        if os.path.exists(f"models/{service}.mar"):
            continue

        print(f"Creating {service}.mar")
        os.system(
            f"torch-model-archiver --model-name '{service}' --version 0.0.1 "
            f"--handler {_path}/{service}_handler.py "
            f"--extra-files {_path}/config.json "
            f"--export-path models/"
        )

    os.system(
        f"cp {_path}/config.properties ./config.properties"
    )

    os.system(
        "torchserve --start --model-store models "
        "--models "
        "bot=llm.mar "
        "speaker=speaker.mar "
        "whisper=whisper.mar "
        "sentence_embedder=sentence_embedder.mar "
        "--foreground "
    )


def process_cli():
    add_cwd_to_syspath()
    print_help()

    arguments = sys.argv
    if len(arguments) > 1:
        command = arguments[1]

        if command == "start":
            start_llm_server()

        else:
            print("Unknown argument.\n")
    else:
        print_help()


def main():
    try:
        process_cli()

    except RuntimeError as e:
        print(e)
        print("WAFL_LLM ended due to the exception above.")
