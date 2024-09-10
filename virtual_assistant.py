import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# GARBAGE CODE

# Step 1: Define the URL of the website to scrape
url = "https://en.wikipedia.org/wiki/Combat_robot"  # Example website; change as needed

def fetch_website_content(url):
    """Fetches and returns text content from the specified URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all(['p', 'div', 'span'])
        if not paragraphs:
            return "No paragraphs found on the page."
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching website content: {e}")
        return ""

def get_relevant_text(content, query):
    """Simple text retrieval: Find the most relevant paragraph based on a keyword query."""
    # This is a very basic example; consider more sophisticated retrieval mechanisms
    paragraphs = content.split('.')
    relevant_text = [para for para in paragraphs if query.lower() in para.lower()]
    return ' '.join(relevant_text[:3])  # Return first 3 relevant paragraphs

def generate_response(retrieved_text, query):
    """Generate a response using an LLM."""
    if not retrieved_text:
        return "I couldn't find any information on that topic."

    # Load a pre-trained model and tokenizer using Hugging Face's pipeline
    model = pipeline("text-generation", model="distilgpt2")  # You can use a more advanced model like GPT-4 if available

    # Generate a response
    response = model(f"Question: {query}\nContext: {retrieved_text}\nAnswer:", max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

def main():
    print("Waiting for input...")
    query = input("Enter your question: ")
    print(f"You entered: {query}")
    
    # Step 2: Fetch content from the website
    content = fetch_website_content(url)
    
    # Step 3: Retrieve relevant text based on the query
    retrieved_text = get_relevant_text(content, query)
    
    # Step 4: Generate a response using an LLM
    response = generate_response(retrieved_text, query)
    print(response)

if __name__ == "__main__":
    main()
