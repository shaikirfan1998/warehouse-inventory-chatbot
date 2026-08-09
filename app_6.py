"""
Warehouse Inventory Chatbot — Streamlit App
--------------------------------------------
Run:
    streamlit run app.py
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_CSV = "data/inventory.csv"
LLM_MODEL   = "google/gemma-2-2b-it"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """You are a warehouse inventory assistant.
Answer ONLY using the context provided. Be concise and use exact numbers from the data.
If the answer is not in the context, say "I don't have that information."
For low-stock items always mention current quantity vs reorder level.

Context:
{context}

Question: {question}

Answer:"""

SAMPLE_QUESTIONS = [
    "Which items are out of stock or low?",
    "List all items in category X.",
    "Show items not sold recently.",
    "Which items have zero availability?",
    "What is the total available count?",
    "Show all items with pack size greater than 1.",
]


# ── Data helpers ───────────────────────────────────────────────────────────────

def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop rows where Part Number is literally "Part Number" (duplicate header rows)
    df = df[df["Part Number"].astype(str).str.strip() != "Part Number"].reset_index(drop=True)

    # Rename columns to clean internal names
    df = df.rename(columns={
        "Part Number":          "item_id",
        "Cat":                  "category",
        "Description":          "item_name",
        "[R]Count":             "quantity",
        "[R]Promo":             "promo",
        "[R]Size":              "size",
        "Packsize":             "pack_size",
        "Supplier Part Number": "supplier_part",
        "Available":            "available",
        "Date Last Sold":       "last_sold",
    })

    # Convert quantity & available to numeric safely
    df["quantity"]  = pd.to_numeric(df["quantity"],  errors="coerce").fillna(0)
    df["available"] = pd.to_numeric(df["available"], errors="coerce").fillna(0)

    # Stock status based on available count
    df["stock_status"] = df["available"].apply(
        lambda x: "🔴 OUT / LOW" if x <= 0 else "🟢 IN STOCK"
    )

    return df


def df_to_documents(df: pd.DataFrame) -> list:
    docs = []
    for _, row in df.iterrows():
        content = (
            f"Part Number: {row['item_id']}. "
            f"Description: {row['item_name']}. "
            f"Category: {row['category']}. "
            f"Count/Quantity: {row['quantity']}. "
            f"Available: {row['available']}. "
            f"Stock status: {row['stock_status']}. "
            f"Pack size: {row['pack_size']}. "
            f"Size: {row['size']}. "
            f"Promo: {row['promo']}. "
            f"Supplier Part Number: {row['supplier_part']}. "
            f"Last sold: {row['last_sold']}."
        )
        docs.append(Document(
            page_content=content,
            metadata={"item_id": str(row["item_id"]), "category": str(row["category"])},
        ))

    # Category summaries
    for cat, grp in df.groupby("category"):
        docs.append(Document(
            page_content=(
                f"Category summary — {cat}: {len(grp)} items, "
                f"total count {grp['quantity'].sum()}, "
                f"total available {grp['available'].sum()}, "
                f"out/low stock: {(grp['stock_status'].str.contains('OUT')).sum()}."
            ),
            metadata={"item_id": "SUMMARY", "category": str(cat)},
        ))

    # Global summary
    docs.append(Document(
        page_content=(
            f"Warehouse overall: {len(df)} items, "
            f"total available {df['available'].sum()}, "
            f"{df['stock_status'].str.contains('OUT').sum()} items out/low stock, "
            f"categories: {', '.join(df['category'].astype(str).unique())}."
        ),
        metadata={"item_id": "GLOBAL"},
    ))
    return docs


# ── Build RAG chain (modern LCEL — no deprecated RetrievalQA) ─────────────────

@st.cache_resource(show_spinner=False)
def build_chain(csv_bytes: bytes, hf_token: str):
    import io
    df   = pd.read_csv(io.BytesIO(csv_bytes))
    df   = enrich_df(df)
    docs = df_to_documents(df)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    splitter    = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits      = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 6})

    hf_client = InferenceClient(
        model=LLM_MODEL,
        token=hf_token,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question: str) -> str:
        # Retrieve relevant chunks
        docs = retriever.invoke(question)
        context = format_docs(docs)

        # Build prompt
        filled_prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # Call HuggingFace Inference API directly
        response = hf_client.chat_completion(
            messages=[{"role": "user", "content": filled_prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    return rag_chain, retriever, df


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Warehouse Inventory Chatbot", page_icon="🏭", layout="wide")
st.title("🏭 Warehouse Inventory Chatbot")
st.caption("Ask questions about your inventory in plain English.")

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Setup")
    hf_token = st.text_input(
        "HuggingFace API Token",
        type="password",
        value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
        help="Get your free token at huggingface.co/settings/tokens",
    )

    st.markdown("---")
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Your inventory CSV", type=["csv"])
    use_sample    = st.checkbox("Use sample data", value=not bool(uploaded_file))

    st.markdown("---")
    st.subheader("Sample questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    st.caption("Stack: LangChain · HuggingFace · FAISS · Streamlit")


# ── Load data ──────────────────────────────────────────────────────────────────

if not hf_token:
    st.warning("Add your HuggingFace API token in the sidebar to get started.")
    st.stop()

if uploaded_file:
    csv_bytes = uploaded_file.read()
elif use_sample and os.path.exists(DEFAULT_CSV):
    with open(DEFAULT_CSV, "rb") as f:
        csv_bytes = f.read()
else:
    st.info("Upload a CSV file or check 'Use sample data' in the sidebar.")
    st.stop()

with st.spinner("Building vector store… (first load takes ~30s)"):
    chain, retriever, df = build_chain(csv_bytes, hf_token)


# ── Metric cards ───────────────────────────────────────────────────────────────

low_stock = df["stock_status"].str.contains("OUT").sum()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total items",     len(df))
col2.metric("Total available", int(df["available"].sum()))
col3.metric("Out / low stock", low_stock, delta=f"-{low_stock}" if low_stock else None, delta_color="inverse")
col4.metric("Categories",      df["category"].nunique())

with st.expander("View inventory table"):
    st.dataframe(
        df[["item_id", "item_name", "category", "quantity", "available",
            "stock_status", "pack_size", "size", "supplier_part", "last_sold"]],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")


# ── Chat ───────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hi! I've loaded **{len(df)} inventory items**. Ask me anything about your stock."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.session_state.pop("pending_question", None) or st.chat_input("Ask about your inventory…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer      = chain(prompt)
            source_docs = retriever.invoke(prompt)

        st.markdown(answer)

        with st.expander(f"Sources ({len(source_docs)} chunks retrieved)"):
            for i, doc in enumerate(source_docs, 1):
                st.caption(f"[{i}] {doc.page_content[:200]}…")

    st.session_state.messages.append({"role": "assistant", "content": answer})
