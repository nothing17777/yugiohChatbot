# Use sentence-transformers locally — runs on your own machine, no per-call cost, no rate limit.
import json

# Load the card data you already fetched and saved from the YGOPRODeck API
with open("all_cards.json") as f:
    cards = json.load(f)

import os
# Sanity check — confirm working directory and that the file actually loaded from there
print(os.getcwd())
print(os.path.exists("all_cards.json"))

from sentence_transformers import SentenceTransformer

# Load a small, free, local embedding model (downloads once, then runs offline)
model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, free, good enough for this


# Turn each card dict into a single text string to embed
def card_to_text(card):
    return f"{card['name']} ({card['type']}): {card['desc']}"

# Build the list of texts, one per card
texts = [card_to_text(c) for c in cards]

# Embed all card texts at once (batched internally for speed)
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

import chromadb

# Create a local, on-disk vector database (no server, no account needed)
client = chromadb.PersistentClient(path="./yugioh_db")
try:
    client.delete_collection("cards")
except:
    pass
collection = client.create_collection("cards")
# Prepare IDs and metadata to store alongside each embedding
ids = [str(c["id"]) for c in cards]
metadatas = [{
    "name": c["name"],
    "type": c["type"],
    "archetype": c.get("archetype", "")
} for c in cards]

# Chroma caps how many items can be added per call, so insert in batches
batch_size = 5000
for i in range(0, len(cards), batch_size):
    collection.add(
        ids=ids[i:i+batch_size],
        documents=texts[i:i+batch_size],
        embeddings=embeddings[i:i+batch_size].tolist(),
        metadatas=metadatas[i:i+batch_size]
    )
    print(f"Added batch {i} to {i+batch_size}")  # progress log so you can see it's not stuck