import json
import os
from typing import Optional

from pydantic import BaseModel

SUMMARY_ABSTRACT = 'summary_abstract'
SUMMARY_EXTRACT = 'summary_extract'

PATH = os.path.dirname(os.path.abspath(__file__)) + '/paper_dataset.json'
#PATH = os.path.dirname(os.path.abspath(__file__)) + '/test.json'

# Models

class Metadata(BaseModel):
    year: int

class Paper(BaseModel):
    metadata: Metadata
    authors: list[str]
    summary: str
    title: str
    claims: Optional[list[str]] = None

    @property
    def year(self) -> int:
        return self.metadata.year

class Sample(BaseModel):
    papers: list[Paper]
    query: str
    summary_abstract: Optional[str] = None
    summary_extract: Optional[str] = None

def load_data(count: int = 2, path: str = PATH) -> list:
    with open(path) as f:
        raw_data = json.load(f)
        return [Sample.parse_obj(d) for d in raw_data[:count]]

def save_data(data: list[Sample], path: str) -> None:
    if not path:
        print('No path provided')
        return

    with open(path, 'w') as f:
        json.dump([sample.dict() for sample in data], f)

if __name__ == '__main__':
    data = load_data()
    print("First query: ", data[0].query)
    print("First paper of first query: ", data[0].papers[0].title)

