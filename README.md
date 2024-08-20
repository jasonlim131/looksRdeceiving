# Visual Sycophancy in Multimodal Models

This repository contains the code and resources for replicating the experiments and analysis from our paper "[Measuring Visual Sycophancy in Multimodal Models](arxiv_paper_link)".

## Table of Contents
1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Setup](#setup)
4. [Generating Experiments](#generating-experiments)
5. [Evaluating Models](#evaluating-models)
6. [Analyzing Results](#analyzing-results)
7. [Models Tested](#models-tested)
8. [Benchmarks](#benchmarks)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

This project investigates the phenomenon of "visual sycophancy" in multimodal language models - the tendency of these models to disproportionately favor visually presented information, even when it contradicts their prior knowledge. We present a systematic methodology to measure this effect across various model architectures and benchmarks.

## Repository Structure

- `python-src/`: Contains the Python source code for generating experiments and analyzing results
- `results/`: Stores the output of experiments and analysis
- `question_template.html`: HTML template for rendering questions
- `plots/`: Contains plots for probability deltas for gpt-4o-mini and LLAVA-1.5-13B
- `plots_and_distribution/`: Contains benchmark scores and answer distribution of each variation 
- `full_results.rtf`: Comprehensive results in Rich Text Format

## Setup

[Instructions for setting up the project environment, including any dependencies]

## Generating Experiments

To generate the experimental prompts:

1. For vMMLU:

python python-src/generate_vmmlu.py 

This will create each rendered prompts in default directory `output_directory/vmmlu_{variation}_rendered/question{i}.png`

2. For vSocialIQa:
python python-src/generate_vsocialiqa.py

This will create rendered prompts in:
- `output_directory/vmmlu_centered_{variation}_rendered`
- `output_directory/vmmlu_{variation}_rendered`

## Evaluating Models

[Instructions for running the evaluations on different models]

## Analyzing Results

[Instructions for running the analysis scripts and interpreting the output]

## Models Tested

We evaluated visual sycophancy in the following models:

- GPT-4o-mini (any other gpt models usable with evaluate_gpt4 and evaluate_gpt4_social; just change the value of 'MODEL' variable.
- Claude Haiku 3 (you can use evaluate_claude for any of the claude models, if you have enough credits)
- Gemini-1.5-flash
- LLAVA-1.5-13B (vMMLU only)

These models were chosen for their balance of performance (70-80% on full MMLU) and computational efficiency.

## Benchmarks

1. Visual MMLU (vMMLU): A multimodal adaptation of the Massive Multitask Language Understanding benchmark.
2. Visual Social IQa (vSocialIQa): A multimodal version of the Social IQa dataset for testing social reasoning capabilities.

## Results

Detailed results can be found in `full_results.rtf`. Key findings include:
- [Summary of main results]
- [Any particularly interesting or surprising findings]

## Contributing

We welcome contributions to this project! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

