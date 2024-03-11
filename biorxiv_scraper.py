import configparser
import dataclasses
import json
from datetime import datetime, timedelta
from html import unescape
from typing import List, Optional, Tuple
import re
import requests

from dataclasses import dataclass, field


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class Paper:
    authors: Tuple[str, ...] = field(default_factory=tuple)
    title: str = ''
    abstract: str = ''
    doi: str = ''
    link: str = ''

    def __post_init__(self):
        # Convert authors list to tuple for immutability, if initiated with a list
        if isinstance(self.authors, list):
            self.authors = tuple(self.authors)

    def __hash__(self):
        return hash(self.doi)

    def __eq__(self, other):
        if not isinstance(other, Paper):
            return False
        return self.doi == other.doi

def get_papers_from_biorxiv_api() -> List[Paper]:
    # look for papers in biorxiv from the last day

    # Get today's date and yesterday's date
    today = datetime.now()
    yesterday = today - timedelta(days=1)

    # Format the dates
    date_format = "%Y-%m-%d"
    today_str = today.strftime(date_format)
    yesterday_str = yesterday.strftime(date_format)

    # Create the URL string
    url = f"https://api.biorxiv.org/details/biorxiv/{yesterday_str}/{today_str}/0/json"

    # Get the number of new papers that day
    response = requests.get(url)
    data = response.json()
    n_papers = data['messages'][0]['total']
    
    # Collect all of the papers
    paper_list = []
    idx = 0
    n_papers_left = n_papers
    while n_papers_left > 0:
        url = f"https://api.biorxiv.org/details/biorxiv/{yesterday_str}/{today_str}/{idx}/json"
        response = requests.get(url)
        data = response.json()
        paper_list.extend(data['collection'])
        idx += 99
        n_papers_left -= 99

    # Convert all papers to Paper format
    paper_set = set([])
    for paper in paper_list:
        PaperObject = Paper(
            authors=paper['authors'].split('; '),
            title=paper['title'],
            abstract=paper['abstract'],
            doi=paper['doi'],
            link=f"https://www.biorxiv.org/content/{paper['doi']}",
        )
        paper_set.add(PaperObject)

    return list(paper_set)



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configs/config.ini")
    paper_list, timestamp, last_id = get_papers_from_biorxiv_rss("cs.CL", config)
    print(timestamp)
    paper_list = get_papers_from_biorxiv_api("cs.CL", timestamp, last_id)
    print([paper.doi for paper in api_paper_list])
    print("success")
