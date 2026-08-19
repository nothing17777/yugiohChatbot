import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Load environment variables (optional, for LangSmith tracking)
load_dotenv()

# Initialize Chroma client (persistent on-disk DB)
client = chromadb.PersistentClient(path="./yugioh_db")
collection = client.get_collection("cards")

# Load embedding model (same as used in the notebook)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize the LLM (the same model used in the notebook)
llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0.3,
)

def retrieve_relevant_cards(query, k=5):
    """Encode the query and retrieve the top‑k most similar card embeddings."""
    query_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
    )
    # Extract documents and their metadata
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    # Build a list of (doc, meta) tuples – doc contains the full text including
    # effect, ATK/DEF, etc. (metadata only stores name/type/archetype/etc.)
    return list(zip(docs, metadatas))


def retrieve_by_archetype(archetype_name, k=30):
    results = collection.query(
        query_embeddings=model.encode([archetype_name]).tolist(),
        n_results=k,
        where={"archetype": archetype_name}
    )
    return list(zip(results["documents"][0], results["metadatas"][0]))


def classify_intent(question: str) -> str:
    prompt = f"""Classify the following question as either "card_question" or "general".
"card_question" = about cards, rules, archetypes, stats, effects, combos.
"general" = greetings, small talk, thanks, chit-chat.
Response must be ONLY "card_question" or "general".
Question: {question}
Answer:"""
    try:
        response = llm.invoke(prompt)
        intent = response.content.strip().lower()
        if "general" in intent:
            return "general"
        return "card_question"
    except:
        return "card_question"


BOT_AVATAR = "assets/bot_avatar.jpg"

# Streamlit UI
st.set_page_config(page_title="Yu-Gi-Oh! RAG Chatbot", page_icon="assets/bot_avatar.jpg", layout="wide")

with st.sidebar:
    st.button("＋ New chat")
    st.button("🔍 Search")
    st.button("💬 History")

st.markdown("""
<style>
/* Centered narrow chat column */
.block-container {
    max-width: 720px !important;
    padding-top: 4.5rem !important;
    padding-bottom: 6rem !important;
    overflow: visible !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Smaller, subtle header */
.app-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.4rem 0 0.3rem 0;
    overflow: visible;
    line-height: 1.4;
}

.app-header img {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: contain;
    background: rgba(255, 255, 255, 0.06);
    flex-shrink: 0;
}

.app-header span {
    font-size: 1.4rem;
    font-weight: 600;
    line-height: 1.5;
    display: inline-block;
    padding-top: 0.15rem;
}

/* Shrink caption */
.stCaption p {
    font-size: 0.75rem !important;
    opacity: 0.5 !important;
}

/* Spacing between chat messages */
.stChatMessage {
    margin-bottom: 1.2rem !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}

/* Assistant message subtle background */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
[data-testid="stChatMessage"]:has(img) {
    background-color: rgba(255, 255, 255, 0.03) !important;
}

/* User message slightly different shade */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: transparent !important;
}

/* Rounded pill-style chat input */
[data-testid="stChatInput"] {
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15) !important;
    overflow: hidden;
}

[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
}

/* Status expander styling */
[data-testid="stStatusWidget"] {
    border-radius: 12px !important;
}

[data-testid="stSidebar"] {
    background-color: #000000;
    width: 220px !important;
}
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: none !important;
    text-align: left !important;
    color: rgba(255,255,255,0.8) !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <img src="app/static/bot_avatar.jpg" alt="Kuriboh">
    <span>Yu-Gi-Oh! RAG Chatbot</span>
</div>
""", unsafe_allow_html=True)
st.caption("Ask anything about Yu-Gi-Oh! cards")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        "<h2 style='text-align:center; margin-top:12vh; opacity:0.85; font-weight:500;'>Where should we begin?</h2>",
        unsafe_allow_html=True
    )

# Display conversation history
for msg in st.session_state.messages:
    avatar = BOT_AVATAR if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Chat input (real-time assistant-style)
prompt = st.chat_input("Ask me about Yu-Gi-Oh! cards")
if prompt:
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate answer
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        intent = classify_intent(prompt)
        
        if intent == "general":
            response = llm.invoke(f"You are a friendly Yu-Gi-Oh card assistant. Respond naturally and briefly to this message: {prompt}")
            answer = response.content
            st.markdown(answer)
            sources = []
        else:
            # Check for archetype/listing request first (Bypass LLM for accuracy)
            if "archetype" in prompt.lower() or "cards in" in prompt.lower():
                with st.status("Thinking...", expanded=True) as status:
                    st.write("🔍 Searching for relevant cards...")
                    sources = retrieve_relevant_cards(prompt, k=5)
                    guessed_archetype = sources[0][1].get("archetype", "")
                    if guessed_archetype:
                        st.write(f"🎯 Archetype identified: **{guessed_archetype}**")
                        sources = retrieve_by_archetype(guessed_archetype, k=30)
                        card_list = "\n".join(f"- **{meta.get('name', 'Unknown')}**" for _, meta in sources)
                        answer = f"Cards in the **{guessed_archetype}** archetype ({len(sources)} found):\n\n{card_list}"
                    else:
                        answer = "I couldn't identify the archetype. Could you be more specific?"
                    status.update(label="Done", state="complete", expanded=False)
                st.markdown(answer)
            else:
                # Standard RAG flow for other questions
                with st.status("Thinking...", expanded=True) as status:
                    st.write("🔍 Searching for relevant cards...")
                    
                    # Check for cached follow-up
                    if st.session_state.get("cached_sources") is not None and (
                        "how many" in prompt.lower() or "total" in prompt.lower() or
                        "those cards" in prompt.lower() or "these cards" in prompt.lower() or
                        len(prompt.strip().split()) <= 3
                    ):
                        sources = st.session_state.cached_sources
                    else:
                        sources = retrieve_relevant_cards(prompt, k=5)
    
                    st.write(f"✅ Found {len(sources)} relevant card(s)")
                    st.write("🤖 Generating answer...")
                    
                    context = "\n\n".join(doc for doc, _ in sources)
                    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:])
                    
                    prompt_text = f"""Answer using ONLY the card names and facts listed in the context below.
    Format your answer as a markdown bulleted list, one card per line, like this:
    - **Card Name** — ATK/DEF, effect summary.
    
    If simply listing cards, just list names cleanly.
    
    Conversation so far:
    {history_text}
    
    Context:
    {context}
    
    Question: {prompt}
    Answer:"""
                    response = llm.invoke(prompt_text)
                    answer = response.content
                    status.update(label="Done", state="complete", expanded=False)
                
                st.markdown(answer)
    
            # Sources expander
            with st.expander("Sources"):
                for doc, meta in sources:
                    st.markdown(f"**{meta.get('name', '')}** ({meta.get('type', '')})")
                    if meta.get("image_url"):
                        st.image(meta["image_url"], width=200)
                    st.write(doc)
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.cached_sources = sources