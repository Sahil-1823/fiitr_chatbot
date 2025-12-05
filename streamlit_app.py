#!/usr/bin/env python3
"""
streamlit_app.py - FITTR AI Assistant UI with LlamaIndex, OpenAI embeddings, and MMR

Features:
- LlamaIndex VectorStoreIndex with OpenAI text-embedding-3-large
- MMR (Maximal Marginal Relevance) for diverse, relevant retrieval
- Reduced retrieval (k=10 candidates, MMR threshold=0.7) for focused results
- Document type badges (research paper vs blog article)
- Clean, user-friendly Streamlit interface
"""

# CRITICAL: Remove problematic environment variables BEFORE any imports
import os
import sys

# Remove the problematic CORS variable from system environment
for key in list(os.environ.keys()):
    if 'CHROMA_SERVER_CORS_ALLOW_ORIGINS' in key.upper():
        print(f"Removing problematic env var: {key}")
        del os.environ[key]

os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "none"

# Import logging utilities
from utils.logger import setup_logger, log_execution_time, PerformanceLogger, log_rag_query

# Now safe to import everything else
import streamlit as st
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import time

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RouterQueryEngine, CustomQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

# Chroma imports
import chromadb
from chromadb.config import Settings as ChromaSettings

# OpenAI for text generation
from openai import OpenAI

load_dotenv()

# Initialize logger
logger = setup_logger(__name__, level="INFO", json_format=False)

# Load API key - works both locally and on Streamlit Cloud
try:
    # Try Streamlit secrets first (for cloud deployment)
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
    if OPENAI_API_KEY:
        logger.info("Loaded API key from Streamlit secrets")
except (FileNotFoundError, AttributeError):
    # Fall back to environment variables (for local development)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    if OPENAI_API_KEY:
        logger.info("Loaded API key from environment variables")

if not OPENAI_API_KEY:
    logger.error("No OpenAI API key found!")
    st.error("⚠️ OpenAI API key not configured. Please add it to Streamlit secrets or .env file.")
    st.stop()

OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
CHROMA_DB_DIR = "chroma_llamaindex_db"

logger.info("Application starting", extra={
    'extra_data': {
        'chroma_db_dir': CHROMA_DB_DIR,
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'api_key_configured': bool(OPENAI_API_KEY)
    }
})

# -----------------------------
#  UI CONFIG
# -----------------------------
st.set_page_config(page_title="FITTR AI Assistant", page_icon="💬", layout="wide")
st.title(" FITTR AI Assistant")
st.caption("Ask anything about health, fitness, nutrition, and FITTR content.")

# -----------------------------
#  LOAD LLAMAINDEX INDEX (Cached)
# -----------------------------
@st.cache_resource
def load_index():
    """Load LlamaIndex VectorStoreIndex from Chroma (cached)."""
    logger.info("Loading VectorStoreIndex from ChromaDB")
    try:
        with PerformanceLogger(logger, "load_index", db_dir=CHROMA_DB_DIR):
            # Setup OpenAI embeddings (must match ingestion)
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-large",
                api_key=OPENAI_API_KEY  # Use the already loaded API key
            )
            
            # Initialize Chroma client
            chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_DIR,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=False,
                    is_persistent=True
                )
            )
            
            # Get the collection
            chroma_collection = chroma_client.get_or_create_collection(
                name="fittr_rag_collection"
            )
            
            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Create index from vector store
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model
            )
            
            logger.info("Index loaded successfully")
            return index
    except Exception as e:
        logger.error(f"Failed to load index", extra={
            'extra_data': {
                'error_type': type(e).__name__,
                'db_dir': CHROMA_DB_DIR
            }
        }, exc_info=True)
        st.error(f"Error loading index from Chroma: {str(e)}")
        st.info(f"Please ensure ingestion has completed and {CHROMA_DB_DIR}/ directory exists.")
        raise


@st.cache_resource
def get_retriever(top_k: int = 5, mmr_threshold: float = 0.85):
    """
    Get retriever from index with MMR enabled (cached).
    
    Args:
        top_k: Number of candidates for MMR (default: 5, reduced for speed)
        mmr_threshold: MMR threshold for diversity (default: 0.85, increased for relevance)
    
    Returns:
        LlamaIndex retriever with MMR enabled
    """
    index = load_index()
    return index.as_retriever(
        similarity_top_k=top_k,
        vector_store_query_mode="mmr",
        vector_store_kwargs={
            "mmr_threshold": mmr_threshold
        }
    )


# -----------------------------
#  SMALL TALK QUERY ENGINE
# -----------------------------
class SmallTalkQueryEngine(CustomQueryEngine):
    """Query engine for greetings and small talk - no RAG needed"""
    
    def custom_query(self, query_str: str):
        """Generate friendly response without retrieval"""
        query_lower = query_str.lower().strip()
        
        # Greetings - more enthusiastic!
        if any(query_lower.startswith(g) for g in ["hi", "hello", "hey", "good morning", "good afternoon"]):
            return "Hey there! 👋 So glad you're here! I'm your FITTR AI buddy, ready to help with all things fitness, nutrition, and wellness. What's on your mind today? 💪✨"
        
        # Thanks - warm and encouraging
        elif any(word in query_lower for word in ["thank", "thanks"]):
            return "Aww, you're so welcome! 🤗 I'm always here to help. Got more questions? Fire away! 💪"
        
        # Goodbye - positive and uplifting
        elif any(word in query_lower for word in ["bye", "goodbye", "see you"]):
            return "Take care! 🌟 Keep crushing those fitness goals! Can't wait to chat again soon! 💪✨"
        
        # Default - friendly and helpful
        else:
            return "Hmm, I'm not quite sure what you're asking! 🤔 Could you rephrase? I'm great at answering stuff like: weight loss tips, muscle building, nutrition advice, workout routines, and more! What would you like to know? "


# -----------------------------
#  CONVERSATION MEMORY HELPER
# -----------------------------
def format_chat_history(messages: List[Dict], max_exchanges: int = 3) -> str:
    """
    Format recent chat history for context.
    
    Args:
        messages: List of message dictionaries from st.session_state.messages
        max_exchanges: Maximum number of user-assistant exchanges to include
    
    Returns:
        Formatted chat history string
    """
    if not messages:
        return "(No previous conversation)"
    
    # Take last N*2 messages (each exchange = user + assistant)
    recent_messages = messages[-(max_exchanges * 2):]
    
    # Filter out small talk to reduce noise
    formatted = []
    for msg in recent_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        route = msg.get("route", "")
        
        # Skip small talk exchanges from history
        # Safely handle route being None or missing
        if route and "small_talk" in str(route).lower():
            continue
        
        if role == "user":
            formatted.append(f"User: {content}")
        elif role == "assistant":
            # Truncate long answers to save tokens
            truncated = content[:200] + "..." if len(content) > 200 else content
            formatted.append(f"Assistant: {truncated}")
    
    return "\n".join(formatted) if formatted else "(No previous conversation)"


@st.cache_resource
def create_router_query_engine():
    """
    Create RouterQueryEngine with LLM-based routing.
    Routes between small_talk (no RAG) and knowledge queries (full RAG).
    Optimized for speed with gpt-4o-mini.
    """
    # Setup fast LLM for routing (gpt-4o-mini for speed)
    routing_llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    
    # Setup answer generation LLM - optimized for speed
    answer_llm = LlamaOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1,  # Lower temperature for faster, more focused responses
         
        api_key=OPENAI_API_KEY
    )
    Settings.llm = answer_llm
    
    # Small talk query engine
    small_talk_engine = SmallTalkQueryEngine()
    
    # Knowledge query engine (RAG-based) with friendly prompt
    index = load_index()
    
    # Create friendly, conversational QA prompt
    from llama_index.core.prompts import PromptTemplate
    
    qa_prompt = PromptTemplate(
        """You're a friendly fitness & health buddy!  
Use the context below to answer in a warm, human style.

Rules:
• Use short paragraphs (2–3 lines each)  
• Add bullet points (•) only when helpful  
• Be friendly, positive, and encouraging  
• If this is a follow-up question based on previous conversation, acknowledge that context naturally
• Do NOT write excessively long answers — keep it moderately detailed  
• Focus only on relevant information from context

Context from Documents:
{context_str}

Question:
{query_str}


Answer (be detailed with bullet points whenever needed • when helpful, stay warm & encouraging):"""
    )
    
    # Optimized for speed: reduced to 3 documents, streaming enabled
    # Note: MMR mode removed due to ChromaDB version compatibility on Streamlit Cloud
    knowledge_engine = index.as_query_engine(
        similarity_top_k=3,  # Reduced from 5 for speed
        text_qa_template=qa_prompt,
        llm=answer_llm,
        # vector_store_query_mode="mmr",
        # vector_store_kwargs={"mmr_threshold": 0.85},
        streaming=True  # Enable streaming for perceived speed
    )
    
    # Create query engine tools
    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=small_talk_engine,
            name="small_talk",
            description="ONLY for greetings ('hi', 'hello'), thanks ('thank you'), or goodbyes ('bye'). NOT for ANY fitness/health questions."
        ),
        QueryEngineTool.from_defaults(
            query_engine=knowledge_engine,
            name="knowledge_query",
            description="""
        Use this for ALL questions related to:
        - fitness, health, nutrition, exercise, weight loss, muscle building
        - FITTR products, FITTR services, FITTR plans, FITTR coaches
        - FITTR app features, FITTR offerings, FITTR content, FITTR pricing
        Any informational question should use this tool, even if it is general or organizational.
"""
        ),
    ]
    
    # Create router with LLM-based selector (gpt-4o-mini for speed)
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=routing_llm),
        query_engine_tools=query_engine_tools,
    )
    
    return router_query_engine


# -----------------------------
#  BUILD CONTEXT FROM NODES
# -----------------------------
def build_context_from_nodes(nodes: List[NodeWithScore]) -> str:
    """
    Build context string from retrieved nodes.
    
    Args:
        nodes: List of NodeWithScore objects
    
    Returns:
        Formatted context string
    """
    text_blocks = []
    for i, node in enumerate(nodes, start=1):
        meta = node.node.metadata or {}
        source = meta.get("source", "unknown")
        title = meta.get("title", "Untitled")
        doc_type = meta.get("document_type", "unknown")
        
        text_blocks.append(
            f"(S{i}) [Source: {source} - {title} - Type: {doc_type}]\n{node.node.text}"
        )
    
    return "\n\n---\n\n".join(text_blocks)


# -----------------------------
#  QUERY PREPROCESSING (LLM-Based)
# -----------------------------
def should_preprocess(query: str) -> bool:
    """Check if query needs preprocessing (has typos/issues)"""
    # Skip if query is very short or a greeting
    if len(query.strip()) < 3 or query.lower().strip() in ['hi', 'hello', 'hey', 'bye', 'thanks', 'thank you']:
        return False
    
    # Common typos that indicate preprocessing is needed
    common_typos = ['hart', 'wieght', 'excersize', 'excercise', 'protien', 'nutrician', 
                    'muscal', 'muscels', 'loosing', 'loose weight', 'caloeries', 'fittness']
    
    query_lower = query.lower()
    return any(typo in query_lower for typo in common_typos)


@st.cache_data(ttl=3600)
def preprocess_query_with_llm(query: str) -> str:
    """
    Use LLM to intelligently correct typos and improve query.
    Only called for queries with obvious issues (smart preprocessing).
    
    Args:
        query: Raw user query
    
    Returns:
        Preprocessed and corrected query
    """
    system_prompt = """Fix typos in this fitness/health query. Be concise.


Examples:
"wieght loss" → "weight loss"
"excersize" → "exercise"

Return ONLY corrected query:"""
    
    try:
        # Use gpt-4o-mini for cost-effective preprocessing
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=100
        )
        
        corrected_query = response.choices[0].message.content.strip()
        return corrected_query
    
    except Exception as e:
        # Fallback to original query if LLM fails
        print(f"Preprocessing error: {e}")
        return query


# -----------------------------
#  GENERATE ANSWER
# -----------------------------
def generate_answer_with_router(query: str, chat_history: str = "") -> Tuple[str, List[NodeWithScore], str, str]:
    """
    Generate answer using LLM-based router with conversation memory.
    
    Args:
        query: User's question
        chat_history: Formatted conversation history
    
    Returns:
        Tuple of (answer, nodes, category, route_name)
    """
    logger.info(f"Processing query", extra={
        'extra_data': {
            'query': query[:100],  # Truncate long queries
            'query_length': len(query),
            'has_history': bool(chat_history and chat_history != "(No previous conversation)")
        }
    })
    
    print(f"\n[Query] {query}")
    timing_start = time.time()
    
    # Smart preprocessing - only if needed
    preprocess_start = time.time()
    if should_preprocess(query):
        preprocessed_query = preprocess_query_with_llm(query)
        preprocess_time = time.time() - preprocess_start
        print(f"[Preprocessed] {preprocessed_query} ({preprocess_time:.2f}s)")
    else:
        preprocessed_query = query
        print(f"[Skipped Preprocessing] Query is clean")
    
    # Get router
    router = create_router_query_engine()
    
    # Check if this is small talk (greetings, thanks, goodbye)
    # We need to check BEFORE adding conversation context, otherwise "hi" won't match startswith()
    query_lower = preprocessed_query.lower().strip()
    is_small_talk = (
        any(query_lower.startswith(g) for g in ["hi", "hello", "hey", "good morning", "good afternoon"]) or
        any(word in query_lower for word in ["thank", "thanks", "bye", "goodbye", "see you"])
    )
    
    # Prepare query with conversation context if available
    # BUT only for non-small-talk queries (small talk doesn't need context)
    query_with_context = preprocessed_query
    if not is_small_talk and chat_history and chat_history != "(No previous conversation)":
        # Prepend chat history to give context to the LLM
        query_with_context = f"[Previous conversation context:\n{chat_history}]\n\nCurrent question: {preprocessed_query}"
        print(f"[Added conversation context ({len(chat_history)} chars)]")
    
    # Query with router (LLM decides which engine to use)
    route_start = time.time()
    
    print(f"[Routing...]", end="", flush=True)
    
    # Add error handling for out-of-domain queries
    try:
        # Note: LlamaIndex doesn't support passing custom context in router.query()
        # The chat_history is included in the QA prompt template above
        response = router.query(query_with_context)
        route_time = time.time() - route_start
    except (IndexError, AttributeError, KeyError, ValueError) as e:
        # Router failed - likely out-of-domain query
        logger.warning("Router query failed - out of domain", extra={
            'extra_data': {
                'query': query[:100],
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        })
        
        print(f" Router Error: {type(e).__name__}")
        route_time = time.time() - route_start
        
        # Return friendly out-of-domain message
        out_of_domain_msg = (
            "Hey! 👋 I'm your FITTR AI Assistant, specialized in **fitness, nutrition, and health** topics. "
            "\n\nI don't have information about that particular topic, but I'd love to help with:\n"
            "• 💪 Fitness & workout routines\n"
            "• 🥗 Nutrition & diet advice\n"
            "• 🏋️ Muscle building & weight loss\n"
            "• 🧘 Health & wellness tips\n"
            "• 📱 FITTR app & services\n\n"
            "What fitness or health question can I help you with today? ✨"
        )
        
        print(f"⏱️ TIMING BREAKDOWN:")
        print(f"   Route: out_of_domain (fallback)")
        print(f"   Total: {route_time:.2f}s")
        
        return out_of_domain_msg, [], "out_of_domain", "fallback"
    
    route_time = time.time() - route_start
    print(f" Done ({route_time:.2f}s)")
    
    # Handle streaming response - return generator for UI
    if hasattr(response, 'response_gen'):
        # Return generator for token-by-token streaming
        print(f"[Streaming enabled]")
        answer_or_gen = response.response_gen
    else:
        # Non-streaming response
        answer_or_gen = str(response)
        print(f"[No streaming]")
    
    # Detect route by checking for source nodes
    nodes_start = time.time()
    nodes = []
    if hasattr(response, 'source_nodes') and response.source_nodes:
        nodes = response.source_nodes
        route_name = "knowledge_query"
    else:
        route_name = "small_talk"
    nodes_time = time.time() - nodes_start
    
    # Return generator or string for streaming display
    # Note: Timing logged after streaming completes in UI
    return answer_or_gen, nodes, "unified", route_name


def generate_answer(query: str, chat_history: str = "") -> Tuple[str, List[NodeWithScore], str]:
    """
    Generate answer using router-based approach with conversation memory.
    
    Args:
        query: User's question
        chat_history: Formatted conversation history
    
    Returns:
        Tuple of (answer, nodes, route_name)
    """
    # Use router-based approach with chat history
    answer, nodes, category, route = generate_answer_with_router(query, chat_history)
    return answer, nodes, route


# -----------------------------
#  GET DOCUMENT TYPE BADGE
# -----------------------------
def get_doc_type_badge(doc_type: str) -> str:
    """
    Get a colored badge for document type.
    
    Args:
        doc_type: Type of document (research_paper or blog_article)
    
    Returns:
        HTML badge string
    """
    if doc_type == "research_paper":
        return '<span style="background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">📄 Research Paper</span>'
    elif doc_type == "blog_article":
        return '<span style="background-color: #2196F3; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">📝 Blog Article</span>'
    else:
        return '<span style="background-color: #9E9E9E; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">📋 Document</span>'



# -----------------------------
#  EAGER LOADING ON STARTUP
# -----------------------------
if 'router_loaded' not in st.session_state:
    with st.spinner("🚀 Loading AI models... This happens once."):
        try:
            logger.info("Initializing AI models")
            # Force load index and router
            _ = load_index()
            _ = create_router_query_engine()
            st.session_state.router_loaded = True
            logger.info("AI models loaded successfully")
            st.success(" AI ready! Ask me anything about fitness, nutrition, and health.", icon="✅")
        except Exception as e:
            logger.error("Failed to load AI models", extra={
                'extra_data': {'error_type': type(e).__name__}
            }, exc_info=True)
            st.error(f" Error loading AI models: {str(e)}")
            st.session_state.router_loaded = False

# -----------------------------
#  CHAT UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Only show evidence for knowledge queries (not small talk)
        if msg["role"] == "assistant" and "nodes" in msg and len(msg["nodes"]) > 0:
            route = msg.get("route", "unknown")
            if "small_talk" not in route.lower():
                with st.expander("📚 Sources"):
                    for i, node in enumerate(msg["nodes"], start=1):
                        meta = node.node.metadata or {}
                        doc_type = meta.get("document_type", "unknown")
                        source_url = meta.get("source_url", "N/A")
                        title = meta.get("title", "Untitled")
                        
                        # Display: badge, title, and source URL
                        st.markdown(f"{i}. {get_doc_type_badge(doc_type)}", unsafe_allow_html=True)
                        st.markdown(f"   **{title}**")
                        st.markdown(f"   Source: {source_url}")
                        if i < len(msg["nodes"]):
                            st.markdown("---")

# Chat input
if query := st.chat_input("💬 Ask me about fitness, nutrition, or health..."):
    # Display user message
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Generate answer with router and conversation history
    with st.spinner(" Thinking..."):
        # Format recent chat history (last 3 exchanges)
        chat_history = format_chat_history(st.session_state.messages, max_exchanges=3)
        answer_or_gen, nodes, route = generate_answer(query, chat_history)
    
    # Display assistant message with streaming
    with st.chat_message("assistant"):
        # Use st.write_stream for token-by-token streaming
        if hasattr(answer_or_gen, '__iter__') and not isinstance(answer_or_gen, str):
            answer = st.write_stream(answer_or_gen)
        else:
            st.markdown(answer_or_gen)
            answer = answer_or_gen
        
        # Only show evidence for knowledge queries
        if len(nodes) > 0 and "small_talk" not in route.lower():
            with st.expander("📚 Sources"):
                for i, node in enumerate(nodes, start=1):
                    meta = node.node.metadata or {}
                    doc_type = meta.get("document_type", "unknown")
                    source_url = meta.get("source_url", "N/A")
                    title = meta.get("title", "Untitled")
                    
                    # Display: badge, title, and source URL
                    st.markdown(f"{i}. {get_doc_type_badge(doc_type)}", unsafe_allow_html=True)
                    st.markdown(f"   **{title}**")
                    st.markdown(f"   Source: {source_url}")
                    if i < len(nodes):
                        st.markdown("---")
    
    # Store message with route information
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "nodes": nodes,
        "route": route
    })

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **FITTR AI Assistant** - Your friendly fitness buddy! 
   
    -  **754 Documents** - Research + blogs
    
    Ask me anything about fitness, nutrition, and health! 
    """)
    
  