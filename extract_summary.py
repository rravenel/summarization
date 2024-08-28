"""
Test effectivenes of extractive summaries to reduce hallucinations.

Baseline: abstractive summaries
GPT 4o mini
Sample size: 200
Hallucination: 18%
Errors: 0 0%
summarize_abstract: Tokens in: 429673	Tokens out: 83548	Total cost: $0.115
detect_abstract_usage: Tokens in: 620757	Tokens out: 200	Total cost: $0.093
Tokens in: 1050430	Tokens out: 83748	Total cost: $0.208
Time taken: 1155s

GPT 4o
Sample size: 50
Hallucination: 14%
Errors: 0 0%
summarize_abstract: Tokens in: 101675	Tokens out: 15667	Total cost: $0.411
detect_abstract_usage: Tokens in: 144274	Tokens out: 50	Total cost: $0.361
Tokens in: 245949	Tokens out: 15717	Total cost: $0.772
Time taken: 467s

Extractive summaries
GPT 4o mini
Sample size: 200
Hallucination: 19%
Errors: 0 0%
extract_claims: Tokens in: 522480	Tokens out: 389555	Total cost: $0.312
summarize_extract: Tokens in: 731823	Tokens out: 80311	Total cost: $0.158
detect_extract_usage: Tokens in: 617523	Tokens out: 200	Total cost: $0.093
Tokens in: 1871826	Tokens out: 470066	Total cost: $0.563
Time taken: 8708s <= laptop went to sleep; extimate ~1.5hrs

GPT 4o for summarization and detection, not extraction
Sample size: 50
Hallucination: 24%
Errors: 0 0%
summarize_extract: Tokens in: 178436	Tokens out: 15235	Total cost: $0.598
detect_extract_usage: Tokens in: 143841	Tokens out: 50	Total cost: $0.360
Tokens in: 322277	Tokens out: 15285	Total cost: $0.959
Time taken: 617s

GPT 4o full
Sample size: 50
Hallucination: 30%
Errors: 1 2%
extract_claims: Tokens in: 124928	Tokens out: 93242	Total cost: $1.245
summarize_extract: Tokens in: 194606	Tokens out: 15208	Total cost: $0.639
detect_extract_usage: Tokens in: 140919	Tokens out: 49	Total cost: $0.353
Tokens in: 460453	Tokens out: 108499	Total cost: $2.236
Time taken: 2880s


"""

import time
from typing import Any, List, Tuple

from data.fave.parse_data import load_data, save_data, Sample, SUMMARY_ABSTRACT, SUMMARY_EXTRACT
from llm import endpoints
from llm.usage import Usage
import calibrate
import util

# FaVe paper prompt, adapted
PROMPT_EXTRACT = """
Below is a summary of a document.
Please extract ALL the claims from the
document.  

A claim is an independent statement of fact
which includes its own context.  Some sentences
may contain multiple claims.  These must be
separated into individual claims. Be sure to
ALWAYS include the FULL CONTEXT of each claim.

You should give your answer as a bulleted list 
separated by "@@@" and start by saying "The claims are:"
"""

# FaVe paper prompt, adapted
PROMPT_SUMMARIZE_EXT = """
You are a research assistant tasked with
summarizing a collection of claims from
a set of papers.

You will be provided with a research 
question and a collection of claims 
that might address this question.

The question will be provided in the
following format:

Question: [question]

The claims will be provided in the
following format:

Author: [author]
Year: [year]
Claim: [claim]

Write a summary of what the claims
collectively say about the research
question.

You must cite the claims in your summary.
You can use the following format:
Author (year)

You will only include the claims that
directly answer our research question,
ignoring other claims that are only
loosely relevant. Remember to include
citations in the final summary. Your
final summary should use varied and
engaging language.
"""

# FaVe paper prompt, adapted
PROMPT_SUMMARIZE_ABS = """
You will be provided with a research 
question and a collection of abstracts 
from papers that might address this question.

The question will be provided in the
following format:

Question: [question]

The abstracts will be provided in the
following format:

Author: [author]
Year: [year]
Abstract: [abstract]

Write a summary of what the papers
collectively say about the research
question.

You must cite the papers in your summary.
You can use the following format:
Author (year)

You will only include the findings that
directly answer our research question,
ignoring other findings that are only
loosely relevant. Remember to include
citations in the final summary. Your
final summary should use varied and
engaging language.
"""

def get_extraction_prompt(summaries: str) -> str:
    return f'{PROMPT_EXTRACT}\n\n{summaries}'

def extract_claims(data: List[Sample]) -> Usage:
    size = len(data)
    total_usage = Usage()

    print('Extract claims...')
    util.print_progress(0, size)
    for i, sample in enumerate(data):
        papers = sample.papers
        for paper in papers:
            prompt = get_extraction_prompt(paper.summary)
        
            response, usage = endpoints.completion('', prompt, response_format = None) # type: ignore
            if not response:
                continue
        
            claims = response[len('The claims are:'):].split('@@@')
            paper.claims = [claim.strip() for claim in claims]

            total_usage.add_tokens(usage.prompt_tokens, usage.completion_tokens)
            
        util.print_progress(i+1, size)
    print()

    return total_usage

def summarize_extract(data: List[Sample]) -> Usage:
    size = len(data)
    total_usage = Usage()

    print('Extractive summarization...')
    util.print_progress(0, size)
    for i, sample in enumerate(data):
        claims = ''
        for paper in sample.papers:
            for claim in paper.claims: # type: ignore
                claims += f'Author: {paper.authors}\nYear: {paper.year}\nClaim: {claim}\n\n'
        
        user_prompt = f'Question: {sample.query}\n\n{claims}'
        
        response, usage = endpoints.completion(PROMPT_SUMMARIZE_EXT, user_prompt, response_format = None) # type: ignore
        if not response:
            continue
        
        sample.summary_extract = response

        total_usage.add_tokens(usage.prompt_tokens, usage.completion_tokens)
        util.print_progress(i+1, size)
    print()

    save_data(data, 'data/fave/50_extractive_summaries_4o.json')

    return total_usage

def summarize_abstract(data: List[Sample]) -> Usage:
    size = len(data)
    total_usage = Usage()

    print('Abstractive summarization...')
    util.print_progress(0, size)
    for i, sample in enumerate(data):
        papers = sample.papers

        abstracts = ''
        for paper in papers:
            abstracts += f'Author: {paper.authors}\nYear: {paper.year}\nAbstract: {paper.summary}\n\n'

        user_prompt = f'Question: {sample.query}\n\n{abstracts}'

        response, usage = endpoints.completion(PROMPT_SUMMARIZE_ABS, user_prompt, response_format = None) # type: ignore
        if not response:
            continue

        sample.summary_abstract = response

        total_usage.add_tokens(usage.prompt_tokens, usage.completion_tokens)
        util.print_progress(i+1, size)
    print()

    save_data(data, 'data/fave/50_abstractive_summaries_4o.json')

    return total_usage

# test for hallucinations
def detect(data: List[Sample], summary_type) -> Usage:
    size = len(data)
    total_usage = Usage()
    count = 0
    err = 0

    print('Detecting hallucinations...')
    util.print_progress(0, size)
    for i, sample in enumerate(data):
        source = ''
        for paper in sample.papers:
            source += f'{paper.summary}\n\n'
        
        user_prompt = calibrate.get_evaluation_prompt(source, getattr(sample, summary_type)) # type: ignore

        response, usage = endpoints.completion(calibrate.EVALUATION_PROMPT, user_prompt, response_format = None) # type: ignore
        if not response:
            err += 1
            continue
        
        if response.lower() == calibrate.YES.lower():
            count += 1
        elif response.lower() == calibrate.NO.lower():
            pass
        else:
            print('Invalid response')
            err += 1
            continue

        total_usage.add_tokens(usage.prompt_tokens, usage.completion_tokens)
        util.print_progress(i+1, size)
    print()

    print(f'Sample size: {len(data)}')
    print(f'Hallucination: {int(100 * count / (len(data) - err))}%')
    print(f'Errors: {err} {int(100 * err / len(data))}%')

    return total_usage


def main() -> None:
    data = load_data(count = 50, path = 'data/fave/50_extractive_summaries_4o.json')
    #data = load_data(count = 50)

    total_usage = Usage()
    #extract_claims_usage = extract_claims(data)
    #summarize_extract_usage = summarize_extract(data)
    #summarize_abstract_usage = summarize_abstract(data)

    #total_usage.add_usage(extract_claims_usage)
    #total_usage.add_usage(summarize_extract_usage)
    #total_usage.add_usage(summarize_abstract_usage)

    detect_extract_usage = detect(data, SUMMARY_EXTRACT)
    #detect_abstract_usage = detect(data, SUMMARY_ABSTRACT)

    total_usage.add_usage(detect_extract_usage)
    #total_usage.add_usage(detect_abstract_usage)

    #print(f'extract_claims: {extract_claims_usage}')
    #print(f'summarize_extract: {summarize_extract_usage}')
    #print(f'summarize_abstract: {summarize_abstract_usage}')
    print(f'detect_extract_usage: {detect_extract_usage}')
    #print(f'detect_abstract_usage: {detect_abstract_usage}')

    print(total_usage)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Time taken: {int(end - start)}s')