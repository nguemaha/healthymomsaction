import asyncio
import json
import os
import nest_asyncio
import pprint
import base64
from io import BytesIO
import pandas as pd
from playwright.async_api import async_playwright
from openai import OpenAI
from PIL import Image
from tabulate import tabulate
from IPython.display import display, HTML, Markdown
from pydantic import BaseModel
from helper import visualizeCourses

from dotenv import load_dotenv

# Load environment variables
HOME_DIR = os.path.expanduser("~")
load_dotenv(f"{HOME_DIR}/.env")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

class WebScraperAgent:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None

    async def init_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-accelerated-2d-canvas",
                "--disable-gpu",
                "--no-zygote",
                "--disable-audio-output",
                "--disable-software-rasterizer",
                "--disable-webgl",
                "--disable-web-security",
                "--disable-features=LazyFrameLoading",
                "--disable-features=IsolateOrigins",
                "--disable-background-networking",
            ],
        )
        self.page = await self.browser.new_page()

    async def scrape_content(self, url):
        if not self.page or self.page.is_closed():
            await self.init_browser()
        await self.page.goto(url, wait_until="load")
        await self.page.wait_for_timeout(2000)  # Wait for dynamic content
        return await self.page.content()

    async def take_screenshot(self, path="screenshot.png"):
        await self.page.screenshot(path=path, full_page=True)
        return path

    async def screenshot_buffer(self):
        screenshot_bytes = await self.page.screenshot(type="png", full_page=False)
        return screenshot_bytes

    async def close(self):
        await self.browser.close()
        await self.playwright.stop()
        self.playwright = None
        self.browser = None
        self.page = None
        

class DeeplearningCourse(BaseModel):
    title: str
    description: str
    presenter: list[str]
    imageUrl: str
    courseURL: str


class DeeplearningCourseList(BaseModel):
    courses: list[DeeplearningCourse]
   
async def process_with_llm(html, instructions, truncate=False):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": f"""
            You are an expert web scraping agent. Your task is to:
            Extract relevant information from this HTML to JSON 
            following these instructions:
            {instructions}
            
            Extract the title, description, presenter, 
            the image URL and course URL for each of 
            all the courses for the deeplearning.ai website

            Return ONLY valid JSON, no markdown or extra text.""",
            },
            {
                "role": "user",
                "content": html[:150000],  # Truncate to stay under token limits
            },
        ],
        temperature=0.1,
        response_format=DeeplearningCourseList,
    )
    return completion.choices[0].message.parsed

async def webscraper(target_url, instructions):
    result = None
    try:
        # Scrape content and capture screenshot
        print("Extracting HTML Content \n")
        html_content = await scraper.scrape_content(target_url)

        print("Taking Screenshot \n")
        screenshot = await scraper.screenshot_buffer()
        # Process content

        print("Processing..")
        result: DeeplearningCourseList = await process_with_llm(
            html_content, instructions, False
        )
        print("\nGenerated Structured Response")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        await scraper.close()
    return result, screenshot

 
scraper = WebScraperAgent()
target_url = "https://www.deeplearning.ai/courses"  # Deeplearning AI courses
base_url = "https://deeplearning.ai"

instructions = """
    Get all the courses
"""
async def main():
    result, screenshot = await webscraper(target_url, instructions)
    return result, screenshot


result, screenshot = asyncio.run(main())


html_content =  scraper.scrape_content(target_url)
screenshot =  scraper.screenshot_buffer()
result =  process_with_llm(
    html_content, instructions, False
)

courses_data = [course.model_dump() for course in result.courses]

result, screenshot = webscraper(target_url, instructions)