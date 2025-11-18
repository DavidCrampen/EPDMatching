This repository is part of the publication: "Development of a BIM-based AI-driven matching tool for LCA datasets" available on: https://link.springer.com/article/10.1007/s43621-025-02203-8

The goal of this repo is to enable you to setup your own automatic EPD matching pipeline based on openai platform: https://platform.openai.com

Getting Started:
To get started follow these instructions:

1. Create an account on https://platform.openai.com
2. Create a new assistant
3. Copy the instructions from Custom\_GPT\_material/Custom\_GPT\_Instruction.txt into the available field (System Instructions) of your assistant
4. Reduce the Temperature parameter to 0, set the Response Format type to JSON-object
5. Upload your vector store (for reference, we provide a chunked version of our dataset of the Ökobaudat from https://www.oekobaudat.de/ in the vector\_store\_oekobaudat\_example folder, additionally we have uploaded the raw data to Examples/Epds
6. Create a new API key
7. Done

Setting up Python environment:
We testet the pipeline on python version 3.11

1. git clone https://github.com/DavidCrampen/EPDMatching.git
2. create new virtual env and run: pip install -r requirements.txt in terminal
3. Afterwards, insert your API key and assistant id into the EPD\_Matching\_pipeline.py skript and you should be able to run the example inputs with the Duplex\_A.ifc, which we used in the paper from the examples.
4. If you want to match your own EPD dataset, you have to change DEFAULT\_EPD\_CONFIG in the EPD\_utils/CO2\_from\_matches\_and\_volumes.py (lines 17-24) with your own column names from the csv.
5. We provide a simple example skript to create json based chunks from a csv file containing the EPDs in CUSTOM\_GPT\_material/CSV\_to\_Chunked\_json.py, however again you will have to adjust the columns names in the columns to the data you want to provide for each EPD.
