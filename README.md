## This is the github repo for replicating the code from {arxiv_paper_link}


We recorded and measured the visual bias / sycophancy in proprietary and open-source models
- gpt4o-mini
- Claude sonnet3 and haiku3
- Gemini-1.5-flash
- LLAVA-1.5-Flash (just vmmlu)

Why we chose these models
They are light, cheap, and high performing version of the popular proprietary sota models (appr. 70-80% on the full mmlu evaluation).

## For vmmlu

use output_directory/vmmlu_{variation}_rendered

## For social_i_qa

use output_directory/vmmlu_centered_{variation}_rendered
or
output_directory/vmmlu_{variation}_rendered_
