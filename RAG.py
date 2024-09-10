# Garage Code 
import wikipediaapi 

# Wikidata API or Wikipedia API: You can use the wikipedia-api Python library to retrieve content from Wikipedia articles.
# Pre-trained models for document embeddings: Since Wikipedia articles are long and complex, you can use sentence embeddings (e.g., sentence-transformers) to convert them into vector representations for more efficient retrieval.
# You can use the wikipedia-api library to fetch relevant articles. For demonstration, we’ll fetch the first few paragraphs of a Wikipedia article based on the query.
# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('RAG experiment (ses439@cornell.edu)', 'en')

# Fetch a Wikipedia page by title or search term
def fetch_wikipedia_article(query):
    page = wiki_wiki.page(query)
    if page.exists():
        # Return the summary or a portion of the article
        return page.summary[:2000]  # Fetch the first 2000 characters for brevity
    else:
        return None

# Example usage
# query = "Combat robotics"
# article_text = fetch_wikipedia_article(query)
# print(article_text)

# ----------------------------------------------------------------------------------------------------------------------------
# To make the retrieval more effective, you can use sentence embeddings to create vector representations of the Wikipedia articles. We'll use the sentence-transformers library to convert Wikipedia article text into embeddings.
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Store embeddings for Wikipedia articles
wiki_corpus = ["Article 1 text", "Article 2 text", "Article 3 text"]
corpus_embeddings = model.encode(wiki_corpus, convert_to_tensor=True)

# Encode query
def encode_query(query):
    return model.encode(query, convert_to_tensor=True)

# Retrieve the most relevant article based on cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_from_wikipedia(query, top_k=1):
    query_embedding = encode_query(query)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)
    top_matches = np.argsort(similarities, axis=1)[0][-top_k:][::-1]
    return [wiki_corpus[i] for i in top_matches]

# Example usage
# retrieved_articles = retrieve_from_wikipedia("autonomous robots in combat")
# print(retrieved_articles)

# ----------------------------------------------------------------------------------------------------------------------------
# Once you retrieve the relevant Wikipedia article(s), you can pass them to the generator for generating a response.
from transformers import pipeline

# Initialize the generator model (using GPT-2 or GPT-3)
generator = pipeline('text-generation', model='gpt-2')

# Generate a response based on the retrieved Wikipedia article
def generate_response_wikipedia(retrieved_docs, prompt):
    context = " ".join(retrieved_docs)[:1024]  # Trim context if too large
    input_prompt = f"{context}\n\n{prompt}"
    response = generator(input_prompt, max_length=200, num_return_sequences=1)
    return response[0]['generated_text']

# Example usage
# prompt = "What are the advantages of autonomous robots in combat?"
# response = generate_response_wikipedia(retrieved_articles, prompt)
# print(response)

# ----------------------------------------------------------------------------------------------------------------------------
# Now you can combine everything into a pipeline that searches Wikipedia, retrieves relevant articles, and generates a response.
def rag_wikipedia_pipeline(query, prompt):
    # Retrieve relevant Wikipedia articles
    article_text = fetch_wikipedia_article(query)
    
    if article_text:
        # Retrieve similar articles using embeddings
        retrieved_articles = retrieve_from_wikipedia(article_text)
        # Generate a response using the retrieved articles
        response = generate_response_wikipedia(retrieved_articles, prompt)
        return response
    else:
        return "No relevant Wikipedia article found."

# Example usage
query = "autonomous robots in combat"
prompt = "What are the advantages of autonomous robots in combat?"
output = rag_wikipedia_pipeline(query, prompt)
print(output)
