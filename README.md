This repository is part of the publication: "Development of a BIM-based AI-driven matching tool for LCA datasets" available on: https://link.springer.com/article/10.1007/s43621-025-02203-8

The goal of this repo is to enable you to setup your own automatic EPD matching pipeline based on openai platform: https://platform.openai.com

Getting Started: 
To get started follow these instructions: 
1. Create an account on https://platform.openai.com
2. Create a new assistant
3. Copy the instructions from Custom_GPT_materials/Custom_GPT_Instruction.txt into the available field (System Instructions) of your assistant
4. Reduce the Temperature parameter to 0, set the Response Format type to JSON-object
5. Upload your vector store (for reference, we provide a chunked version of our dataset from https://www.oekobaudat.de/ in  
