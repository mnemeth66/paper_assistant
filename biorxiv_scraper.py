import configparser
import dataclasses
import json
from datetime import datetime, timedelta
from html import unescape
from typing import List, Optional
import re
import requests

from dataclasses import dataclass


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class Paper:
    __slots__ = ['_authors', '_title', '_abstract', '_doi', '_link']  # Prevents dynamic attribute assignment
    
    def __init__(self, authors: List[str], title: str, abstract: str, doi: str, link: str):
        self._authors = tuple(authors)  # Make authors immutable
        self._title = title
        self._abstract = abstract
        self._doi = doi
        self._link = link

    @property
    def authors(self):
        return self._authors

    @property
    def title(self):
        return self._title

    @property
    def abstract(self):
        return self._abstract

    @property
    def doi(self):
        return self._doi

    @property
    def link(self):
        return self._link

    def __hash__(self):
        return hash(self._doi)  # Ensure this matches with the immutable identifier

    def __eq__(self, other):
        if not isinstance(other, Paper):
            return False
        return self._doi == other._doi  # Ensuring equality checks are based on a unique property


def is_earlier(ts1, ts2):
    # compares two biorxiv ids, returns true if ts1 is older than ts2
    return int(ts1.replace(".", "")) < int(ts2.replace(".", ""))


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
