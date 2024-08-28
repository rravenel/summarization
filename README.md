# summarization
Experiments in LLM summarization

## Extractive summarization

Can extracting claims from source text reduce hallucentions in summaries?

We first test our detection process on the HaluEval summarization dataset and found it to have an accuracy of 74% on 1000 samples (equal split positive and negative).

Next, we perform abstractive summarization on the full FaVe dataset (200 samples) and then evaluate the hallucination rate (% summaries with hallucinations).

Finally, we perform extractive summarization on FaVe followed by the same evaluation.  The extractive summarization process consists of extracting claims from the source material, and then summarizing the claims.

We found that the abstractive summarization process has a detected hallucination rate of 18%, and the extractive process has a rate of 19%.  

All steps used GPT 4o mini.

HaluEval dataset: https://github.com/RUCAIBox/HaluEval
FaVe dataset: https://github.com/elicit/fave-dataset
