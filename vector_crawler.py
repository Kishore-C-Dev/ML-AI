import os
import asyncio
import openai
import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from openai import ChatCompletion,AsyncOpenAI
from datetime import datetime, timezone
from urllib.parse import urlparse
import nest_asyncio
import json
import numpy as np
from typing import List, Dict, Any
from xml.etree import ElementTree

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize Hugging Face embeddings
hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=hf_model_name)

# Initialize Chroma vector store
chroma_persist_directory = "./chroma_data"
vectorstore = Chroma(
    collection_name="webpage_content",
    embedding_function=hf_embeddings,
    persist_directory=chroma_persist_directory
)

print(f"HF Vectorstore created with {vectorstore._collection.count()} documents")



# OpenAI API client
openai_api_key = os.getenv("OPENAI_API_KEY")

if os.path.exists(chroma_persist_directory):
    Chroma(persist_directory=chroma_persist_directory, embedding_function=hf_embeddings).delete_collection()
    print("deleted existing collection")

async def get_title_and_summary(markdown: str, url: str) -> dict:
    """Extract title and summary using OpenAI."""
    system_prompt = """You are an AI that extracts titles and summaries from webpage markdown.
    Return a JSON object with 'title' and 'summary' keys.
    Title: Short and descriptive of the page's content.
    Summary: A concise summary of the main points."""

    try:
        # Use OpenAI to generate title and summary
        response= await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{markdown[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        print(f"Response: {response.choices[0].message.content}")
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error with OpenAI: {e}")
        return {"title": "Untitled", "summary": "No summary available"}


async def process_and_store(url: str, markdown: str):
    """Process markdown and store in Chroma vector store."""
    try:
        # Get title and summary
        extracted = await get_title_and_summary(markdown, url)

        # Metadata for the document
        metadata = {
            "url": url,
            "source": "web",
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path,
        }

        # Create a Document object
        document = Document(
            page_content=markdown,
            metadata={**metadata, "title": extracted["title"], "summary": extracted["summary"]},
        )
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents([document])

        documents = [
        Document(
            page_content=chunk.page_content,
            metadata={"source": url, "chunk_number": i}
        )
        for i, chunk in enumerate(chunks)
        ]


        # Add document to Chroma vector store
        vectorstore.add_documents(documents)
        #vectorstore.persist()
        collection = vectorstore._collection
        
        hf_sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        print(f"Embedding Length: {len(hf_sample_embedding)}")
        print(f"Sample Embedding: {hf_sample_embedding[:10]}")  
        
        print(collection.get(limit=1, include=["embeddings"])["embeddings"][0].shape)  
        sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        print(f"length:{len(sample_embedding)}")
        print(f"Processed and stored: {url}")

    except Exception as e:
        print(f"Error processing {url}: {e}")


async def crawl_webpages(urls: list, max_concurrent: int = 5):
    """Crawl multiple webpages and process their content."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    async def process_url(url):
        try:
            result = await crawler.arun(url=url, config=crawl_config, session_id="session1")
            if result.success:
                print(f"Successfully crawled: {url}")
                await process_and_store(url, result.markdown_v2.raw_markdown)
            else:
                print(f"Failed to crawl {url}: {result.error_message}")
        except Exception as e:
            print(f"Error crawling {url}: {e}")

    # Limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_process_url(url):
        async with semaphore:
            await process_url(url)

    try:
        await asyncio.gather(*[limited_process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_all_urls(sitemap_url) -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    #sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    # Example list of URLs to crawl
    #urls = ["https://ai.pydantic.dev", "https://ai.pydantic.dev/logfire/"]

    urls=get_all_urls("https://ai.pydantic.dev/sitemap.xml")

    print(f"Starting to crawl {len(urls)} URLs...")
    await crawl_webpages(urls)
    print("Crawling and processing complete.")


if __name__ == "__main__":
    asyncio.run(main())

"""query = "What is PydanticAI?"
results = vectorstore.similarity_search(query, k=5)

# Display results
for i, doc in enumerate(results):
    print(f"Result {i + 1}:")
    print(f"Text: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")"""


