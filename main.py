from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv() 
print("Loaded weather key:", os.getenv("OPENWEATHER_API_KEY"))



@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations."""
    print("calculator tool has been called")
    return f"The sum of {a} and {b} is {a + b}"


@tool
def multiplier(a: float, b: float) -> str:
    """Useful for multiplying two numbers."""
    print("multiplier tool has been called")
    return f"The product of {a} and {b} is {a * b}"


@tool
def word_counter(text: str) -> str:
    """Useful for counting the number of words in a given text."""
    print("word_counter tool has been called")
    count = len(text.split())
    return f"The text contains {count} words."


@tool
def web_scraper(url: str) -> str:
    """Useful for extracting text content from a webpage."""
    print("web_scraper tool has been called")
    import requests
    from bs4 import BeautifulSoup

    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:1000] + "..." if text else "No text found on the page."
    except Exception as e:
        return f"Error scraping the site: {str(e)}"


@tool
def pdf_summarizer(path: str) -> str:
    """Summarizes the content of a local PDF file given its file path."""
    print("pdf_summarizer tool has been called")
    from PyPDF2 import PdfReader

    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        if len(text.strip()) < 100:
            return "The PDF doesn't contain enough text to summarize."
        return f"Summary: {text[:700]}..."
    except Exception as e:
        return f"Failed to summarize PDF: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Fetches current weather info for a city using OpenWeatherMap."""
    print("get_weather tool has been called")
    import requests

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API key not found in environment."

    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        )
        res = requests.get(url)
        data = res.json()

        if data.get("cod") != 200:
            return f"Failed to get weather: {data.get('message', 'Unknown error')}"

        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"The current weather in {city} is '{weather}' with a temperature of {temp}°C."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


def main():
    model = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    
    tools = [
        calculator,
        word_counter,
        multiplier,
        web_scraper,
        pdf_summarizer,
        get_weather,
    ]

    agent_executor = create_react_agent(model, tools)

    print("Welcome! I’m your assistant.")
    print("You can ask me anything, or type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        print("\nAssistant: ", end="")
        response = ""
        for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}):
            if "message" in chunk:
                response += chunk["message"].content
            elif "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    response += message.content
        print(response)


if __name__ == "__main__":
    main()
