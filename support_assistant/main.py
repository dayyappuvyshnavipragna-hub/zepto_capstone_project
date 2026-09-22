import os
from pathlib import Path
from typing import TypedDict, Literal

import chromadb
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from langgraph.graph import StateGraph, END

ROOT = Path(__file__).resolve().parent
DB = ROOT / "chroma_db"

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(DB))
collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)

KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours"
]

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    answer: str
    sources: list[str]
    confidence: float


# Required structured prompt for the optional real-LLM extension.
PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer-support assistant.

CONTEXT:
Use only the retrieved Zepto policy context supplied below.

TASK:
Answer the customer's question using the retrieved policy context.

FORMAT:
Return JSON with answer, sources, and confidence.

LENGTH:
Keep the answer concise and directly relevant.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.

FEW-SHOT EXAMPLE:
Question: What is the standard delivery fee below INR 149?
Context: Orders below INR 149 incur a flat INR 25 delivery fee.
Answer: {"answer":"Orders below INR 149 incur a flat INR 25 delivery fee.","sources":["doc_01"],"confidence":1.0}

RETRIEVED CONTEXT:
{context}

CUSTOMER QUESTION:
{query}
"""


def classify_intent(state: GraphState):
    q = state["query"].lower()
    if any(k in q for k in KEYWORDS):
        intent = "policy_question"
    else:
        intent = "general_question"

    # In MOCK_LLM mode no LLM call occurs.
    # The MOCK_LLM=0 extension can replace this branch with an LLM call.
    return {"intent": intent}

def call_optional_llm(prompt: str, max_retries: int = 3) -> str:
    """
    Optional real-LLM extension point with retry-on-failure logic.
    The default submission remains offline with MOCK_LLM=1.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            # Optional extension:
            # Replace this section with the selected real-LLM API call.
            return (
                "MOCK_LLM=0 is an optional extension. "
                "The structured prompt is ready for a real LLM call."
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Optional LLM call failed after {max_retries} attempts: {last_error}"
    )


def retrieve_and_answer(state: GraphState):
    query = state["query"]
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    documents = result["documents"][0]
    ids = result["ids"][0]

    top_chunk = documents[0]
    snippet = top_chunk[:200].replace("\n", " ")

    if MOCK_LLM:
        answer = f"Based on the retrieved context: {snippet}"
        confidence = 1.0
    else:
          context = "\n\n".join(documents)
          prompt = PROMPT_TEMPLATE.format(
              context=context,
              query=query
    )

    answer = call_optional_llm(prompt, max_retries=3)
    confidence = 0.5

    return {
        "answer": answer,
        "sources": ids,
        "confidence": confidence
    }


def direct_answer(state: GraphState):
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        answer = "Optional real-LLM extension point for general questions."
        confidence = 0.5

    return {
        "answer": answer,
        "sources": [],
        "confidence": confidence
    }


def route(state: GraphState):
    return state["intent"]


builder = StateGraph(GraphState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

builder.set_entry_point("classify_intent")
builder.add_conditional_edges(
    "classify_intent",
    route,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()

app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = graph.invoke({"query": request.query})

    # Mock mode deterministically creates validated structured output.
    response = AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 1.0)
    )
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)