import json
import os
import sys
import shutil

from wafl_llm.configuration import configuration_names_dict
from wafl_llm.variables import get_variables

_path = os.path.dirname(__file__)
_running_path = os.getcwd()


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
    services = ["llm", "sentence_embedder", "whisper", "speaker", "entailer", "configuration"]
    if os.path.exists("models"):
        print("Removing the prior models/ directory.")
        shutil.rmtree("models/")

    os.system(f"mkdir -p models")

    log_dir = f"{_running_path}/logs/"
    if os.path.exists(log_dir):
        print("Removing the prior logs/ directory.")
        shutil.rmtree(log_dir)

    config_path = f"{_path}/config.json"
    if os.path.exists("config.json"):
        print("Found existing config.json in local directory.")
        config_path = f"{_running_path}/config.json"

    for service in services:
        if os.path.exists(f"models/{service}.mar"):
            continue

        print(f"Creating {service}.mar")
        os.system(
            f"torch-model-archiver --model-name '{service}' --version 0.0.1 "
            f"--handler {_path}/{service}_handler.py "
            f"--extra-files {config_path} "
            f"--export-path models/"
        )

    os.system(f"cp {_path}/config.properties ./config.properties")

    to_run = "torchserve --start --model-store models " "--foreground " "--models "
    to_run += configuration_names_dict["configuration_model"] + " "
    for service, name in json.load(open(config_path)).items():
        if not name:
            print(f"Not running the service named {service}.")
            continue
        if service not in configuration_names_dict:
            print(f"Skipping the unknown service named {service}.")
            continue
        to_run += configuration_names_dict[service] + " "
    print("Executing >", to_run)
    os.system(to_run)


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


if __name__ == "__main__":
    main()
