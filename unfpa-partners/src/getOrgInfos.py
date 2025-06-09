import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# import openai
import requests
from pydantic import BaseModel
from typing import Optional

# Load environment variables
HOME_DIR = os.path.expanduser("~")
load_dotenv(f"{HOME_DIR}/.env")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


def get_org_names():
    df = pd.read_csv("../data/unfpa_partners.csv")
    return df.OrgName.to_list()


# === Pydantic model for structured result ===
class NGOInfo(BaseModel):
    address: Optional[str]
    website: Optional[str]


def search_ngos_serpapi(ngo_name):
    url = "https://serpapi.com/search"
    params = {"q": ngo_name, "api_key": SERPAPI_API_KEY, "num": 5}
    response = requests.get(url, params=params)
    data = response.json()

    # Combine snippets
    snippets = []
    for result in data.get("organic_results", []):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        snippets.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")

    return "\n\n".join(snippets)


def extract_ngo_info_with_gpt1(ngo_name, snippets_text):
    # openai.api_key = OPENAI_API_KEY
    prompt = f"""
        I searched online for the NGO: "{ngo_name}". Below are the search result summaries:
        
        {snippets_text}
        
        Please extract the most likely:
        1. Address of the NGO (or city/country if exact address not available)
        2. Official website URL
        
        If not found, say "Not found".
        return a dictionary with keys: 'address' and 'website'.
        """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.beta.chat.completions.parse(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response  # .choices[0].message.parsed


def extract_ngo_info_with_gpt(ngo_name, snippets_text) -> NGOInfo:
    prompt = f"""
    I searched online for the NGO: "{ngo_name}". Below are the search result summaries:

    {snippets_text}

    Please extract the most likely:
    1. Address of the NGO (or city/country if exact address not available)
    2. Official website URL

    If not found, say "Not found".
    Return a JSON object with keys: 'address' and 'website'.
    """

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format=NGOInfo,
    )

    # Extract structured dict and parse with Pydantic
    parsed_dict = response.choices[0].message.content
    if isinstance(parsed_dict, str):
        import json

        parsed_dict = json.loads(parsed_dict)

    return NGOInfo(**parsed_dict)


def main(ngo_name):
    snippets = search_ngos_serpapi(ngo_name)
    result = extract_ngo_info_with_gpt(ngo_name, snippets)
    infos = result.model_dump()
    return infos


if __name__ == "__main__":
    # Load NGO names from CSV
    org_names = get_org_names()

    # Initialize dictionaries to store results
    websites = {}
    addresses = {}

    # Process each NGO name
    for org_name in org_names:
        print(f"Processing: {org_name}")
        try:
            info = main(org_name)
            websites[org_name] = info["website"]
            addresses[org_name] = info["address"]
        except Exception as e:
            print(f"Error processing {org_name}: {e}")
            websites[org_name] = "Error"
            addresses[org_name] = "Error"

    # Save results to CSV
    print("Saving results to CSV...")

    df = pd.read_csv("../data/unfpa_partners.csv")
    df["URL"] = df["OrgName"].map(websites)
    df["Address"] = df["OrgName"].map(addresses)
    df.to_csv("../data/unfpa_partners-v1.csv", index=False)
