from typing import List
import os
import base64
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
HOME_DIR = os.path.expanduser("~")
load_dotenv(f"{HOME_DIR}/.env")
DATA_DIR = "../data"


# Step 1: Define row-oriented structure
class TableRow(BaseModel):
    Country: str
    OrgName: str
    OrgType: str
    Description: str
    Amount: str


class TableData(BaseModel):
    rows: List[TableRow]


# Step 2: OpenAI client
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Step 3: Encode image
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# Step 4: Prompt builder
def build_vision_prompt(prompt: str, base64_image: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        }
    ]


# Step 5: Prompt text
def get_default_prompt() -> str:
    return """
        You are a data extraction assistant. Your task is to extract structured tabular data from an image of a table.
        Each row in the image has five fields:
            - Country
            - Partner Organization Name
            - Organization Type
            - Project Description
            - Amount (USD)

        Return a JSON array of dictionaries where each dictionary represents a row with these keys.
        Make sure all rows have the same structure. Fix OCR issues. If uncertain, mark data with [inferred].
        Do not guess numerical values. Only return the JSON in the correct format.
    """


# Step 6: Run GPT model and convert to DataFrame
def parse_table_data_from_image(image_path: str, model_name="gpt-4o") -> pd.DataFrame:
    client = get_openai_client()
    base64_img = encode_image_to_base64(image_path)
    prompt = get_default_prompt()
    messages = build_vision_prompt(prompt, base64_img)

    completion = client.beta.chat.completions.parse(
        model=model_name, messages=messages, temperature=0, response_format=TableData
    )

    return completion.choices[0].message.parsed


def main():
    for i in range(1, 46):
        print(f"Processing file: p{i}.png")
        try:
            image_path = f"../data/unfpa_partners/p{i}.png"
            response = parse_table_data_from_image(image_path)
            df = pd.DataFrame([row.model_dump() for row in response.rows])

            df.to_csv(f"../data/unfpa_partners/p{i}.csv", index=False)

        except Exception as e:
            print(f"Error processing file: p{i}.png. Exception: {e}")
            continue


def collectResults():
    datas = []

    for i in range(1, 46):
        df = pd.read_csv(f"../data/unfpa_partners/p{i}.csv")
        datas.append(df)
    data = pd.concat(datas)
    data.to_csv("../data/unfpa_partners.csv", index=False)


if __name__ == "__main__":
    # Example usage
    image_path = "../data/unfpa_partners/p1.png"
    df = parse_table_data_from_image(image_path)
    print(df)
    # Convert to DataFrame if needed
    df = pd.DataFrame([row.dict() for row in df.rows])
    print(df.head())
