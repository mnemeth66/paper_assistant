import configparser
import dataclasses
import json
import os
import re
from typing import List, Tuple, Dict

from anthropic import Anthropic
import retry
from openai import OpenAI
from tqdm import tqdm

from arxiv_scraper import Paper
from arxiv_scraper import EnhancedJSONEncoder
from google import genai


def filter_by_author(all_authors, papers, author_targets, config):
    # filter and parse the papers
    selected_papers = {}  # pass to output
    all_papers = {}  # dict for later filtering
    sort_dict = {}  # dict storing key and score

    # author based selection
    for paper in papers:
        all_papers[paper.doi] = paper
        for author in paper.authors:
            if author in all_authors:
                for alias in all_authors[author]:
                    if alias["authorId"] in author_targets:
                        selected_papers[paper.doi] = {
                            **dataclasses.asdict(paper),
                            **{"COMMENT": "Author match"},
                        }
                        sort_dict[paper.doi] = float(
                            config["SELECTION"]["author_match_score"]
                        )
                        break
    return selected_papers, all_papers, sort_dict


def filter_papers_by_hindex(all_authors, papers, config):
    # filters papers by checking to see if there's at least one author with > hcutoff hindex
    paper_list = []
    for paper in papers:
        max_h = 0
        for author in paper.authors:
            if author in all_authors:
                max_h = max(
                    max_h, max([alias["hIndex"] for alias in all_authors[author]])
                )
        if max_h >= float(config["FILTERING"]["hcutoff"]):
            paper_list.append(paper)
    return paper_list


def calc_price(model, usage):
    if model == "gpt-4-0125-preview":
        return (0.01 * usage.prompt_tokens + 0.03 * usage.completion_tokens) / 1000.0
    if model == "gpt-4":
        return (0.03 * usage.prompt_tokens + 0.06 * usage.completion_tokens) / 1000.0
    if (model == "gpt-3.5-turbo") or (model == "gpt-3.5-turbo-0125"):
        return (0.0005 * usage.prompt_tokens + 0.0015 * usage.completion_tokens) / 1000.0
    if (model == "claude-3-sonnet-20240229"):
        # Anthropic uses input and output tokens
        return (0.003 * usage.input_tokens + 0.015 * usage.output_tokens) / 1000.0
    if (model == "claude-3-haiku-20240307"):
        return (0.00025 * usage.input_tokens + 0.00125 * usage.output_tokens) / 1000.0


@retry.retry(tries=3, delay=10)
def call_client(full_prompt, config):
    client_type = config["SELECTION"]["model_provider"]
    model = config["SELECTION"]["model"]

    if client_type == "openai":
        # Set up client
        OAI_KEY = os.environ.get("OAI_KEY")
        if OAI_KEY is None:
            raise ValueError(
                "OpenAI key is not set - please set OAI_KEY to your OpenAI key"
            )
        client = OpenAI(api_key=OAI_KEY)

        # Get response
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0,
            seed=0,
        )
        content, usage = response.choices[0].message.content, response.usage
    elif client_type == "anthropic":
        # Set up client
        ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
        if ANTHROPIC_API_KEY is None:
            raise ValueError(
                "ANTHROPIC API Key is not set."
            )
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # Get response
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0,

        )
        content, usage = message.content[0].text, message.usage
    else:
        print(f'client type {client_type} not yet implemented.')
    
    return content, usage


def run_and_parse_chatgpt(full_prompt, config):
    # just runs the chatgpt prompt, tries to parse the resulting JSON
    content, usage = call_client(full_prompt, config)
    out_text = content
    out_text = re.sub("```jsonl\n", "", out_text)
    out_text = re.sub("```", "", out_text)
    out_text = re.sub(r"\n+", "\n", out_text)
    out_text = re.sub("},", "}", out_text).strip()
    # split out_text line by line and parse each as a json.
    json_dicts = []
    for line in out_text.split("\n"):
        # try catch block to attempt to parse json
        try:
            json_dicts.append(json.loads(line))
        except Exception as ex:
            if config["OUTPUT"].getboolean("debug_messages"):
                print("Exception happened " + str(ex))
                print("Failed to parse LM output as json")
                print(out_text)
                print("RAW output")
                print(content)
            continue
    return json_dicts, calc_price(config["SELECTION"]["model"], usage)


def paper_to_string(paper_entry: Paper) -> str:
    # renders each paper into a string to be processed by GPT
    new_str = (
        "DOI: "
        + paper_entry.doi
        + "\n"
        + "Title: "
        + paper_entry.title
        + "\n"
        + "Authors: "
        + " and ".join(paper_entry.authors)
        + "\n"
        + "Abstract: "
        + paper_entry.abstract[:4000]
    )
    return new_str


def batched(items, batch_size):
    # takes a list and returns a list of list with batch_size
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def filter_papers_by_title(
    papers, config, base_prompt, criterion
) -> List[Paper]:
    filter_postfix = 'Identify any papers that are absolutely and completely irrelavent to the criteria, and you are absolutely sure your friend will not enjoy, formatted as a list of DOIs like ["DOI1", "DOI2", "DOI3"..]. Be extremely cautious, and if you are unsure at all, do not add a paper in this list. You will check it in detail later.\n Directly respond with the list, do not add ANY extra text before or after the list. Even if every paper seems irrelevant, please keep at least TWO papers'
    batches_of_papers = batched(papers, 20)
    final_list = []
    cost = 0
    for batch in batches_of_papers:
        papers_string = "".join([paper_to_titles(paper) for paper in batch])
        full_prompt = (
            base_prompt + "\n " + criterion + "\n" + papers_string + filter_postfix
        )
        model = config["SELECTION"]["model"]
        content, usage = call_client(full_prompt, config)
        cost += calc_price(model, usage)
        out_text = content
        try:
            filtered_set = set(json.loads(out_text))
            for paper in batch:
                if paper.doi not in filtered_set:
                    final_list.append(paper)
                else:
                    print("Filtered out paper " + paper.doi)
        except Exception as ex:
            print("Exception happened " + str(ex))
            print("Failed to parse LM output as list " + out_text)
            print(content)
            continue
    return final_list, cost


def paper_to_titles(paper_entry: Paper) -> str:
    return "DOI: " + paper_entry.doi + " Title: " + paper_entry.title + "\n"

def parse_gemini_response(response_text: str) -> List[Dict]:
    """
    Robust parsing of Gemini's response to extract JSON objects.
    
    Args:
        response_text (str): Full text response from Gemini
    
    Returns:
        List[Dict]: Parsed JSON responses
    """
    # Split the text into lines
    lines = response_text.strip().split('\n')
    
    json_responses = []
    for line in lines:
        try:
            # Try to parse each line as JSON
            # Remove any leading/trailing whitespace
            cleaned_line = line.strip()
            
            # Skip empty lines
            if not cleaned_line:
                continue
            
            # Try to parse the line as JSON
            json_obj = json.loads(cleaned_line)
            json_responses.append(json_obj)
        
        except json.JSONDecodeError:
            # If a line can't be parsed, print it for debugging
            print(f"Could not parse line as JSON: {line}")
            # Optionally, you could log this or handle it differently
            continue
    
    return json_responses

def filter_by_gpt(papers: List[Paper], base_prompt: str, criterion: str, postfix_prompt: str, config: configparser.ConfigParser) -> Tuple[Dict[str, Paper], Dict[str, float], float]:
    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Initialize result containers
    selected_papers = {}
    sort_dict = {}
    total_cost = 0.0
    
    # Process papers in batches of 10
    batches = batched(papers, 10)  # Using the existing batched function
    
    for batch in tqdm(batches, desc="Processing paper batches"):
        # Format prompt for scoring papers
        prompt = f"{base_prompt}\n{criterion}\n\n[PAPERS]\n"
        for paper in batch:
            prompt += f"DOI: {paper.doi}\nTitle: {paper.title}\nAbstract: {paper.abstract}\n\n"
        prompt += postfix_prompt
        
        try:
            print(f"Processing batch of {len(batch)} papers...")
            print(f"Full prompt length: {len(prompt)} characters")
            
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt
            )
            
            # Debug: print full response text
            print("Full response text:")
            print(response.text)
            print("-" * 50)
            
            # Parse response into JSON format
            json_responses = parse_gemini_response(response.text)
            
            print(f"Parsed {len(json_responses)} JSON responses")
            
            # Process responses for this batch
            for paper, json_response in zip(batch, json_responses):
                # Calculate total score
                total_score = json_response.get('RELEVANCE', 0) + json_response.get('NOVELTY', 0)
                
                print(f"Paper DOI: {paper.doi}, Total Score: {total_score}")
                
                # Only include papers with non-zero scores
                if total_score > 0:
                    selected_papers[paper.doi] = {
                        'paper': paper,
                        'relevance': json_response.get('RELEVANCE', 0),
                        'novelty': json_response.get('NOVELTY', 0)
                    }
                    sort_dict[paper.doi] = total_score
                else:
                    print(f"Paper {paper.doi} filtered out due to zero total score")
        
        except Exception as e:
            print(f"Error processing batch: {e}")
            print(f"Problematic prompt: {prompt[:1000]}...")  # Print first 1000 chars of prompt
            # Continue to next batch instead of returning immediately
            continue

    
    # Return after processing all batches
    print(f"Total selected papers: {len(selected_papers)}")
    return selected_papers, sort_dict, total_cost


def run_on_batch(
    papers: List[Paper],
    base_prompt: str,
    criterion: str,
    postfix_prompt: str,
    config: configparser.ConfigParser,
) -> Tuple[List[Dict], float]:
    """Run the model on a batch of papers."""
    if config["SELECTION"]["model_provider"] == "gemini":
        return filter_by_gpt(papers, base_prompt, criterion, postfix_prompt, config)
    else:
        papers_string = "".join([paper_to_string(paper) for paper in papers])
        full_prompt = base_prompt + "\n " + criterion + "\n" + papers_string + postfix_prompt
        return run_and_parse_chatgpt(full_prompt, config)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configs/config.ini")
    # now load the api keys
    keyconfig = configparser.ConfigParser()
    keyconfig.read("configs/keys.ini")
    # S2_API_KEY = keyconfig["KEYS"]["semanticscholar"]  # Commented out as not needed
    client = OpenAI(api_key=keyconfig["KEYS"]["openai"])
    # deal with config parsing
    with open("configs/base_prompt.txt", "r") as f:
        base_prompt = f.read()
    with open("configs/paper_topics.txt", "r") as f:
        criterion = f.read()
    with open("configs/postfix_prompt.txt", "r") as f:
        postfix_prompt = f.read()
    # loads papers from 'in/debug_papers.json' and filters them
    with open("in/debug_papers.json", "r") as f:
        # with open("in/gpt_paper_batches.debug-11-10.json", "r") as f:
        paper_list_in_dict = json.load(f)
    papers = [
        [
            Paper(
                doi=paper["doi"],
                authors=paper["authors"],
                title=paper["title"],
                abstract=paper["abstract"],
            )
            for paper in batch
        ]
        for batch in paper_list_in_dict
    ]
    all_papers = {}
    paper_outputs = {}
    sort_dict = {}
    total_cost = 0
    for batch in tqdm(papers):
        json_dicts, cost = run_on_batch(
            batch, base_prompt, criterion, postfix_prompt, client, config
        )
        total_cost += cost
        for paper in batch:
            all_papers[paper.doi] = paper
        for jdict in json_dicts:
            paper_outputs[jdict["DOI"]] = {
                **dataclasses.asdict(all_papers[jdict["DOI"]]),
                **jdict,
            }
            sort_dict[jdict["DOI"]] = jdict["RELEVANCE"] + jdict["NOVELTY"]

        # sort the papers by relevance and novelty
    print("total cost:" + str(total_cost))
    keys = list(sort_dict.keys())
    values = list(sort_dict.values())

    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    sorted_keys = [keys[idx] for idx in argsort(values)[::-1]]
    selected_papers = {key: paper_outputs[key] for key in sorted_keys}

    with open(
        config["OUTPUT"]["output_path"] + "filter_paper_test.debug.json", "w"
    ) as outfile:
        json.dump(selected_papers, outfile, cls=EnhancedJSONEncoder, indent=4)
