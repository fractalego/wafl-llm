import os
import sys

from wafl_llm.variables import get_variables


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
    pass ####


def process_cli():
    add_cwd_to_syspath()
    print_help()

    arguments = sys.argv
    if len(arguments) > 1:
        command = arguments[1]

        if command == "start_llm":
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
