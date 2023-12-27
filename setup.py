from wafl_llm.variables import get_variables
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="wafl_llm",
    version=get_variables()["version"],
    url="http://github.com/fractalego/wafl_llm",
    author="Alberto Cetoli",
    author_email="alberto@fractalego.io",
    description="A hybrid chatbot - LLM side.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[
        "wafl_llm",
    ],
    package_data={
        "wafl_llm": [
            "config.json",
            "config.properties",
        ],
    },
    install_requires=[
        "torch == 2.0.0",
        "transformers == 4.35.2",
        "sentencepiece == 0.1.98",
        "accelerate == 0.20.3",
        "bitsandbytes == 0.41.1",
        "optimum == 1.8.6",
        "onnx == 1.13.0",
        "datasets == 2.8.0",
        "deepspeed == 0.9.2",
        "mpi4py == 3.1.4",
        "torchserve == 0.7.1",
        "torch-model-archiver == 0.7.1",
        "torch-workflow-archiver == 0.2.6",
        "nvgpu == 0.9.0",
        "fairseq == 0.12.2",
        "g2p_en == 2.1.0",
        "sentence_transformers == 2.2.2",
        "einops == 0.6.1",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    entry_points={
        "console_scripts": ["wafl_llm=wafl_llm.command_line:main"],
    },
    include_package_data=True,
    zip_safe=False,
)
