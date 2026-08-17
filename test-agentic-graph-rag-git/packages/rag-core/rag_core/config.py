"""Agentic Graph RAG configuration via Pydantic Settings."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# Runtime environment
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
RUNTIME_ENV_FILE = CONFIG_DIR / "runtime.env"
RELATIONSHIP_SCHEMA_FILE = CONFIG_DIR / "relationship_schema.json"
PROMPTS_FILE = CONFIG_DIR / "prompts.json"


# ============================================================================
# Prompt loading
# ============================================================================


def _load_prompts() -> dict:
    """Load system prompts from JSON file."""
    if not PROMPTS_FILE.exists():
        return {}

    try:
        with open(PROMPTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("openai", {})
    except Exception:
        return {}


def _get_prompt(key: str, default: str) -> str:
    """Get prompt value from JSON file or use default."""
    prompts = _load_prompts()
    return prompts.get(key, default)


# ============================================================================
# Neo4j
# ============================================================================


class Neo4jSettings(BaseSettings):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"

    model_config = SettingsConfigDict(
        env_file=RUNTIME_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="NEO4J_",
        extra="ignore",
    )


# ============================================================================
# OpenAI / LLM
# ============================================================================


class OpenAISettings(BaseSettings):
    api_key: str = ""
    base_url: str = ""

    # Separate embedding endpoint configuration
    embedding_api_key: str = ""
    embedding_base_url: str = ""

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    router_model: str = "deepseek-v4-flash"
    cypher_model: str = "deepseek-v4-flash"
    corrector_model: str = "deepseek-v4-flash"
    synthesis_model: str = "deepseek-v4-flash"

    # Model used specifically for relationship discovery.
    # Usually the cheapest capable model should be selected here.
    relationship_discovery_model: str = "deepseek-v4-flash"

    llm_temperature: float = 0.0

    # ========================================================================
    # System Prompts (loaded from config/prompts.json)
    # ========================================================================

    router_system_prompt: str = Field(
        default_factory=lambda: _get_prompt(
            "router_system_prompt",
            "You are a routing agent. Analyze the user query and route it to "
            "the appropriate component or retrieve relevant information from "
            "the knowledge graph.",
        )
    )

    cypher_system_prompt: str = Field(
        default_factory=lambda: _get_prompt(
            "cypher_system_prompt",
            "You are a Neo4j Cypher query generator. Convert natural language "
            "queries into accurate Cypher queries. Ensure the queries are "
            "syntactically correct and optimized.",
        )
    )

    corrector_system_prompt: str = Field(
        default_factory=lambda: _get_prompt(
            "corrector_system_prompt",
            "You are a Cypher query corrector. Review and fix incorrect or "
            "suboptimal Cypher queries. Provide explanations for corrections.",
        )
    )

    synthesis_system_prompt: str = Field(
        default_factory=lambda: _get_prompt(
            "synthesis_system_prompt",
            "You are a helpful assistant that synthesizes information from "
            "multiple sources into clear, accurate, and well-structured answers. "
            "Always cite your sources.",
        )
    )

    decomposer_system_prompt: str = Field(
        default_factory=lambda: _get_prompt(
            "decomposer_system_prompt",
            "You are a search query decomposer for a RAG system. "
            "Given a query, generate different search sub-queries that together "
            "cover ALL aspects of the original question. Each sub-query should focus "
            "on a DIFFERENT section, component, or angle. For enumeration queries "
            "(list all, describe all), each sub-query should target a DIFFERENT item "
            "from the expected list.",
        )
    )

    entity_extraction_system_prompt: str = Field(
        default_factory=lambda: _get_prompt(
            "entity_extraction_system_prompt",
            "You are an entity extraction expert. "
            "Extract important entities and direct relationships explicitly "
            "supported by the text. Do not invent entities. Do not invent relationships. "
            "Prefer concrete domain-important concepts such as entities, "
            "objects, processes, structures, symptoms, products, people, "
            "organizations, technical concepts, locations, events, etc.",
        )
    )

    relationship_discovery_system_prompt: str = Field(
        default_factory=lambda: _get_prompt(
            "relationship_discovery_system_prompt",
            "You are a precise knowledge graph relationship discovery system. "
            "Return only valid JSON.",
        )
    )

    model_config = SettingsConfigDict(
        env_file=RUNTIME_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="OPENAI_",
        extra="ignore",
    )


# ============================================================================
# Indexing
# ============================================================================


class IndexingSettings(BaseSettings):
    # ------------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------------

    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ------------------------------------------------------------------------
    # Graph indexing
    # ------------------------------------------------------------------------

    skeleton_beta: float = 0.25
    knn_k: int = 10
    pagerank_damping: float = 0.85

    # ------------------------------------------------------------------------
    # Semantic Relations
    # ------------------------------------------------------------------------

    semantic_relations_enabled: bool = True
    semantic_relation_types: str = ""

    semantic_relation_batch_size: int = 25
    semantic_relation_max_pairs: int = 60
    semantic_relation_min_cooccurrence: int = 1
    semantic_relation_min_confidence: float = 0.65
    semantic_relation_context_chars: int = 1200
    semantic_relation_max_per_pair: int = 4

    semantic_relation_model: str = ""
    semantic_relation_temperature: float = 0.0

    # ------------------------------------------------------------------------
    # Corpus Discovery
    # ------------------------------------------------------------------------

    discovery_llm_enabled: bool = True
    discovery_max_pairs: int = 50
    discovery_batch_size: int = 25
    discovery_sample_contexts: int = 3
    discovery_min_evidence: int = 1

    model_config = SettingsConfigDict(
        env_file=RUNTIME_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="INDEXING_",
        extra="ignore",
    )


# ============================================================================
# Relationship Discovery
# ============================================================================


class RelationshipDiscoverySettings(BaseSettings):
    """
    Configuration for automatic discovery of semantic relationships.

    Discovery intentionally does NOT require a predefined relationship
    dictionary.

    It can work in three modes:

    - heuristic
        No LLM. Finds entities that occur together and extracts the
        words between them as candidate relationship phrases.

    - llm
        LLM extracts semantic relationship candidates.

    - both
        Runs both methods and combines their results.
    """

    enabled: bool = True

    mode: str = "both"

    # Maximum number of documents used during one discovery run.
    max_documents: int = 10

    # Maximum number of chunks sampled from every document.
    # This is the main cost-control parameter for LLM discovery.
    max_chunks_per_document: int = 5

    # Number of chunks sent in one LLM request.
    batch_size: int = 5

    # Maximum number of relationship candidates shown after aggregation.
    max_candidates: int = 100

    # Candidate must occur at least this many times.
    # Keep the default at 1 so valid relationships are not filtered out
    # during discovery on small corpora.
    min_frequency: int = 1

    # Maximum number of words between two entity mentions
    # for heuristic relation extraction.
    context_window: int = 12

    # Maximum number of LLM relationship candidates per chunk.
    max_relationships_per_chunk: int = 20

    # If true, already approved relationships are supplied to the LLM
    # as normalization hints.
    use_approved_schema: bool = True

    # Path to the human-approved relationship schema.
    schema_path: str = str(RELATIONSHIP_SCHEMA_FILE)

    model_config = SettingsConfigDict(
        env_file=RUNTIME_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="RELATIONSHIP_DISCOVERY_",
        extra="ignore",
    )


# ============================================================================
# Retrieval
# ============================================================================


class RetrievalSettings(BaseSettings):
    top_k_vector: int = 10
    top_k_final: int = 10
    vector_threshold: float = 0.5
    max_hops: int = 3
    ppr_alpha: float = 0.15

    model_config = SettingsConfigDict(
        env_file=RUNTIME_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="RETRIEVAL_",
        extra="ignore",
    )


# ============================================================================
# Agent
# ============================================================================


class AgentSettings(BaseSettings):
    max_retries: int = 2
    relevance_threshold: float = 2.0

    model_config = SettingsConfigDict(
        env_file=RUNTIME_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="AGENT_",
        extra="ignore",
    )


# ============================================================================
# Root Settings
# ============================================================================


class Settings(BaseSettings):
    neo4j: Neo4jSettings = Field(
        default_factory=Neo4jSettings
    )

    openai: OpenAISettings = Field(
        default_factory=OpenAISettings
    )

    indexing: IndexingSettings = Field(
        default_factory=IndexingSettings
    )

    relationship_discovery: RelationshipDiscoverySettings = Field(
        default_factory=RelationshipDiscoverySettings
    )

    retrieval: RetrievalSettings = Field(
        default_factory=RetrievalSettings
    )

    agent: AgentSettings = Field(
        default_factory=AgentSettings
    )

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=RUNTIME_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ============================================================================
# Cached settings
# ============================================================================


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Create and cache application settings from runtime.env."""
    return Settings()


# ============================================================================
# OpenAI client
# ============================================================================


def make_openai_client(
    settings: Settings | None = None,
    for_embedding: bool = False,
):
    """Create OpenAI client with optional OpenAI-compatible proxy.
    
    Args:
        settings: Configuration settings
        for_embedding: If True, use embedding endpoint config (separate base_url/api_key)
    """

    from openai import OpenAI

    cfg = settings or get_settings()

    # Select which endpoint config to use
    if for_embedding:
        api_key = cfg.openai.embedding_api_key or cfg.openai.api_key
        base_url = cfg.openai.embedding_base_url or cfg.openai.base_url
    else:
        api_key = cfg.openai.api_key
        base_url = cfg.openai.base_url

    if not api_key and not base_url:
        endpoint = "embedding" if for_embedding else "general LLM"
        raise ValueError(
            f"OpenAI credentials for {endpoint} endpoint not configured. "
            "Set OPENAI_API_KEY or OPENAI_BASE_URL for general LLM, or "
            "OPENAI_EMBEDDING_API_KEY and OPENAI_EMBEDDING_BASE_URL for embeddings."
        )

    kwargs: dict[str, str] = {}

    if api_key:
        kwargs["api_key"] = api_key
    elif base_url:
        kwargs["api_key"] = "none"

    if base_url:
        kwargs["base_url"] = base_url
        
    print(
    "========== OPENAI CLIENT =========="
    )
    print(
        "for_embedding:",
        for_embedding,
    )
    print(
        "api_key configured:",
        bool(api_key),
    )
    print(
        "base_url:",
        base_url,
    )
    print(
        "embedding_model:",
        cfg.openai.embedding_model,
    )
    print(
        "==================================="
    )

    return OpenAI(**kwargs)
