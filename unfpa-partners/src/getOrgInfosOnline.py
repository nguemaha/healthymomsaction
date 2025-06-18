import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# import openai
# import requests
from pydantic import BaseModel
from typing import Optional

# Load environment variables
HOME_DIR = os.path.expanduser("~")
load_dotenv(f"{HOME_DIR}/.env")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
DATADIR = os.path.join(
    HOME_DIR, "Documents", "repos", "healthymomsaction", "unfpa-partners", "data"
)


def get_org_data():
    fname = os.path.join(DATADIR, "unfpa_partners.csv")
    df = pd.read_csv(fname)

    return df


# === Pydantic model for structured result ===
class NGOInfo(BaseModel):
    city: Optional[str]
    address: Optional[str]
    website: Optional[str]
    description: Optional[str]


def find_ngo_infos(ngo_name) -> NGOInfo:
    prompt = f"""
    You are an information extraction assistant. 
    Given the NGO name and country (in the format: name, country), search the web and provide:
    - Full street address (if available) 
    - City where the NGO is located 
    - Official website URL
    - Brief description of the NGO (if available)

    If any item isn’t found, set its value to "Not found".

    **Response format must be:
      - Address
      - City
      - Website URL
      - Description
    **
    Do not include any other information or explanations.
    
    Here is the NGO name: {ngo_name}
    """

    model_input = [
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": prompt,
        },
    ]

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=model_input,
        tools=[{"type": "web_search_preview"}],
        text_format=NGOInfo,
        temperature=0,
    )

    return response.output_parsed


def test(ngo_name):
    ngo_name = "Prime Foundation, Pakistan"
    infos = find_ngo_infos(ngo_name)
    return infos


if __name__ == "__main__":
    from tqdm import tqdm

    df = get_org_data()
    df = df[df.OrgType == "NGO"]
    # Initialize dictionaries to store results
    addresses = {}
    cities = {}
    websites = {}
    descriptions = {}

    # Process each NGO name
    for name, country in tqdm(zip(df.OrgName, df.Country)):
        org_name = f"{name}, {country}"
        # print(f"Processing: {org_name}")

        try:
            info = find_ngo_infos(org_name)

            addresses[name] = info.address
            cities[name] = info.city
            websites[name] = info.website
            descriptions[name] = info.description

        except Exception as e:
            print(f"Error processing {name}: {e}")
            websites[name] = "Not found"
            addresses[name] = "Not found"
            cities[name] = "Not found"
            descriptions[name] = "Not found"

    # Save results to CSV
    print("Saving results to CSV...")
    df["Address"] = df["OrgName"].map(addresses)
    df["City"] = df["OrgName"].map(cities)
    df["URL"] = df["OrgName"].map(websites)
    df["Description"] = df["OrgName"].map(descriptions)
    df.to_csv(os.path.join(DATADIR, "unfpa_partners-ngos.csv"), index=False)
