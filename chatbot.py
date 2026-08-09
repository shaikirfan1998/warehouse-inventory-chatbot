"""
Warehouse Inventory Chatbot — HuggingFace Edition
---------------------------------------------------
Stack (all free):
  - LangChain                  : orchestration
  - HuggingFace Inference API  : free LLM via API (no local download)
  - HuggingFace Embeddings     : free local embeddings
  - FAISS                      : free local vector store
  - Pandas                     : CSV loading & enrichment

Setup:
  1. Get a free HF token → https://huggingface.co/settings/tokens
  2. Paste it in .env as HUGGINGFACEHUB_API_TOKEN=hf_...
  3. pip install -r requirements.txt
  4. python chatbot.py
"""

import os
from dotenv import load_dotenv
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

CSV_PATH    = "data/inventory.csv"

# Free HuggingFace Inference API models (no GPU needed):
LLM_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"   # best free option
# Alternatives:
# "HuggingFaceH4/zephyr-7b-beta"
# "google/flan-t5-large"               ← lightest, fastest

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # runs locally, free


# ── 1. Load & enrich CSV ──────────────────────────────────────────────────────

def load_inventory(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["total_value"]  = df["quantity"] * df["unit_price"]
    df["stock_status"] = df.apply(
        lambda r: "LOW STOCK" if r["quantity"] <= r["reorder_level"] else "ADEQUATE",
        axis=1,
    )
    return df


# ── 2. CSV rows → LangChain Documents ────────────────────────────────────────

def df_to_documents(df: pd.DataFrame) -> list[Document]:
    docs = []

    # One Document per item row
    for _, row in df.iterrows():
        content = (
            f"Item ID: {row['item_id']}. "
            f"Name: {row['item_name']}. "
            f"Category: {row['category']}. "
            f"Quantity: {row['quantity']} {row['unit']}. "
            f"Reorder level: {row['reorder_level']}. "
            f"Stock status: {row['stock_status']}. "
            f"Unit price: ${row['unit_price']:.2f}. "
            f"Total value: ${row['total_value']:.2f}. "
            f"Supplier: {row['supplier']}. "
            f"Location: {row['location']}. "
            f"Last updated: {row['last_updated']}."
        )
        docs.append(Document(
            page_content=content,
            metadata={"item_id": row["item_id"], "category": row["category"]},
        ))

    # One summary Document per category
    for cat, grp in df.groupby("category"):
        docs.append(Document(
            page_content=(
                f"Category summary — {cat}: "
                f"{len(grp)} items, "
                f"total quantity {grp['quantity'].sum()}, "
                f"total value ${grp['total_value'].sum():.2f}, "
                f"low-stock count: {(grp['stock_status']=='LOW STOCK').sum()}."
            ),
            metadata={"item_id": "SUMMARY", "category": cat},
        ))

    # Global warehouse summary
    docs.append(Document(
        page_content=(
            f"Warehouse overall: {len(df)} items, "
            f"total value ${df['total_value'].sum():.2f}, "
            f"{(df['stock_status']=='LOW STOCK').sum()} items below reorder level, "
            f"categories: {', '.join(df['category'].unique())}."
        ),
        metadata={"item_id": "GLOBAL"},
    ))

    return docs


# ── 3. Build FAISS vector store ───────────────────────────────────────────────

def build_vectorstore(docs: list[Document]):
    print("Loading embedding model (first run ~90MB download)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    splitter    = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits      = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    print(f"  {len(splits)} chunks indexed into FAISS.")
    return vectorstore


# ── 4. Build RAG chain ────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """<s>[INST]
You are a warehouse inventory assistant.
Answer ONLY using the context provided. Be concise and use exact numbers from the data.
If the answer is not in the context, say "I don't have that information."
For low-stock items always mention current quantity vs reorder level.

Context:
{context}

Question: {question}
[/INST]
Answer:"""

def build_chain(vectorstore):
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found. Add it to your .env file.")

    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        max_new_tokens=512,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


# ── 5. Chat loop ──────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "Which items are low in stock?",
    "What is the total inventory value?",
    "List all items supplied by SafeGuard.",
    "Which category has the highest total value?",
    "What items are stored in Aisle-B?",
    "How many items need reordering?",
]

def chat(chain):
    print("\n" + "═" * 55)
    print("  Warehouse Inventory Chatbot  (type 'quit' to exit)")
    print("═" * 55)
    print("\nSample questions:")
    for q in SAMPLE_QUESTIONS:
        print(f"  • {q}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = chain.invoke({"query": user_input})
        answer = result["result"] if isinstance(result, dict) else result
        print(f"Bot: {answer}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading inventory...")
    df   = load_inventory(CSV_PATH)
    print(f"  {len(df)} items loaded.")

    docs        = df_to_documents(df)
    vectorstore = build_vectorstore(docs)
    chain       = build_chain(vectorstore)

    chat(chain)


if __name__ == "__main__":
    main()
