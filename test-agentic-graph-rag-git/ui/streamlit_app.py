"""
Agentic Graph RAG — Streamlit UI.

8 tabs:

- Ingest
- Search & Q&A
- Graph Explorer
- Discovery
- Agent Trace
- Benchmark
- Reasoning
- Settings

The Discovery tab is intentionally separate from normal ingestion:
load the whole document collection first, then run one corpus-level
semantic relation pass over the existing graph.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

from rag_core.i18n import get_translator


# ============================================================================
# Paths
# ============================================================================

APP_ROOT = Path(__file__).resolve().parent.parent

RUNTIME_ENV_PATH = (
    APP_ROOT
    / "config"
    / "runtime.env"
)

BASE_ENV_PATH = APP_ROOT / ".env"


# ============================================================================
# Runtime environment keys
# ============================================================================

RUNTIME_ENV_KEYS = [
    # ------------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------------

    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_EMBEDDING_API_KEY",
    "OPENAI_EMBEDDING_BASE_URL",
    "OPENAI_ROUTER_MODEL",
    "OPENAI_CYPHER_MODEL",
    "OPENAI_CORRECTOR_MODEL",
    "OPENAI_SYNTHESIS_MODEL",
    "OPENAI_RELATIONSHIP_DISCOVERY_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_DIMENSIONS",
    "OPENAI_LLM_TEMPERATURE",

    # ------------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------------

    "INDEXING_CHUNK_SIZE",
    "INDEXING_CHUNK_OVERLAP",
    "INDEXING_SKELETON_BETA",
    "INDEXING_KNN_K",
    "INDEXING_PAGERANK_DAMPING",

    # ------------------------------------------------------------------------
    # Semantic Relations
    # ------------------------------------------------------------------------

    "INDEXING_SEMANTIC_RELATIONS_ENABLED",
    "INDEXING_SEMANTIC_RELATION_TYPES",
    "INDEXING_SEMANTIC_RELATION_BATCH_SIZE",
    "INDEXING_SEMANTIC_RELATION_MAX_PAIRS",
    "INDEXING_SEMANTIC_RELATION_MIN_COOCCURRENCE",
    "INDEXING_SEMANTIC_RELATION_MIN_CONFIDENCE",
    "INDEXING_SEMANTIC_RELATION_CONTEXT_CHARS",
    "INDEXING_SEMANTIC_RELATION_MAX_PER_PAIR",
    "INDEXING_SEMANTIC_RELATION_MODEL",
    "INDEXING_SEMANTIC_RELATION_TEMPERATURE",

    # ------------------------------------------------------------------------
    # Corpus Discovery
    # These fields belong to IndexingSettings.
    # ------------------------------------------------------------------------

    "INDEXING_DISCOVERY_LLM_ENABLED",
    "INDEXING_DISCOVERY_MAX_PAIRS",
    "INDEXING_DISCOVERY_BATCH_SIZE",
    "INDEXING_DISCOVERY_SAMPLE_CONTEXTS",
    "INDEXING_DISCOVERY_MIN_EVIDENCE",

    # ------------------------------------------------------------------------
    # Relationship Discovery
    # These fields belong to RelationshipDiscoverySettings.
    # ------------------------------------------------------------------------

    "RELATIONSHIP_DISCOVERY_ENABLED",
    "RELATIONSHIP_DISCOVERY_MODE",
    "RELATIONSHIP_DISCOVERY_MAX_DOCUMENTS",
    "RELATIONSHIP_DISCOVERY_MAX_CHUNKS_PER_DOCUMENT",
    "RELATIONSHIP_DISCOVERY_BATCH_SIZE",
    "RELATIONSHIP_DISCOVERY_MAX_CANDIDATES",
    "RELATIONSHIP_DISCOVERY_MIN_FREQUENCY",
    "RELATIONSHIP_DISCOVERY_CONTEXT_WINDOW",
    "RELATIONSHIP_DISCOVERY_MAX_RELATIONSHIPS_PER_CHUNK",
    "RELATIONSHIP_DISCOVERY_USE_APPROVED_SCHEMA",
    "RELATIONSHIP_DISCOVERY_SCHEMA_PATH",

    # ------------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------------

    "RETRIEVAL_TOP_K_VECTOR",
    "RETRIEVAL_TOP_K_FINAL",
    "RETRIEVAL_VECTOR_THRESHOLD",
    "RETRIEVAL_MAX_HOPS",
    "RETRIEVAL_PPR_ALPHA",

    # ------------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------------

    "AGENT_MAX_RETRIES",
    "AGENT_RELEVANCE_THRESHOLD",
]


# ============================================================================
# Known embedding dimensions
# ============================================================================

EMBEDDING_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


# ============================================================================
# Environment helpers
# ============================================================================

def load_runtime_environment() -> None:
    """Load runtime.env into os.environ."""

    if RUNTIME_ENV_PATH.exists():
        load_dotenv(
            dotenv_path=RUNTIME_ENV_PATH,
            override=True,
        )


def _runtime_env_template() -> str:
    """Build runtime.env contents from current environment."""

    values = {
        key: os.environ.get(key, "")
        for key in RUNTIME_ENV_KEYS
    }

    sections = [
        "# Agentic Graph RAG — Runtime Settings",
        "# Managed by Streamlit. Base .env is not modified.",
        "",

        # --------------------------------------------------------------------
        # OpenAI
        # --------------------------------------------------------------------

        "# OpenAI",
        f"OPENAI_API_KEY={values['OPENAI_API_KEY']}",
        f"OPENAI_BASE_URL={values['OPENAI_BASE_URL']}",
        f"OPENAI_EMBEDDING_API_KEY={values['OPENAI_EMBEDDING_API_KEY']}",
        f"OPENAI_EMBEDDING_BASE_URL={values['OPENAI_EMBEDDING_BASE_URL']}",
        f"OPENAI_ROUTER_MODEL={values['OPENAI_ROUTER_MODEL']}",
        f"OPENAI_CYPHER_MODEL={values['OPENAI_CYPHER_MODEL']}",
        f"OPENAI_CORRECTOR_MODEL={values['OPENAI_CORRECTOR_MODEL']}",
        f"OPENAI_SYNTHESIS_MODEL={values['OPENAI_SYNTHESIS_MODEL']}",
        f"OPENAI_RELATIONSHIP_DISCOVERY_MODEL={values['OPENAI_RELATIONSHIP_DISCOVERY_MODEL']}",
        f"OPENAI_EMBEDDING_MODEL={values['OPENAI_EMBEDDING_MODEL']}",
        f"OPENAI_EMBEDDING_DIMENSIONS={values['OPENAI_EMBEDDING_DIMENSIONS']}",
        f"OPENAI_LLM_TEMPERATURE={values['OPENAI_LLM_TEMPERATURE']}",
        "",

        # --------------------------------------------------------------------
        # Indexing
        # --------------------------------------------------------------------

        "# Indexing",
        f"INDEXING_CHUNK_SIZE={values['INDEXING_CHUNK_SIZE']}",
        f"INDEXING_CHUNK_OVERLAP={values['INDEXING_CHUNK_OVERLAP']}",
        f"INDEXING_SKELETON_BETA={values['INDEXING_SKELETON_BETA']}",
        f"INDEXING_KNN_K={values['INDEXING_KNN_K']}",
        f"INDEXING_PAGERANK_DAMPING={values['INDEXING_PAGERANK_DAMPING']}",
        "",

        # --------------------------------------------------------------------
        # Semantic Relations
        # --------------------------------------------------------------------

        "# Semantic Relations",
        (
            "INDEXING_SEMANTIC_RELATIONS_ENABLED="
            f"{values['INDEXING_SEMANTIC_RELATIONS_ENABLED']}"
        ),
        (
            "INDEXING_SEMANTIC_RELATION_TYPES="
            f"{values['INDEXING_SEMANTIC_RELATION_TYPES']}"
        ),
        (
            "INDEXING_SEMANTIC_RELATION_BATCH_SIZE="
            f"{values['INDEXING_SEMANTIC_RELATION_BATCH_SIZE']}"
        ),
        (
            "INDEXING_SEMANTIC_RELATION_MAX_PAIRS="
            f"{values['INDEXING_SEMANTIC_RELATION_MAX_PAIRS']}"
        ),
        (
            "INDEXING_SEMANTIC_RELATION_MIN_COOCCURRENCE="
            f"{values['INDEXING_SEMANTIC_RELATION_MIN_COOCCURRENCE']}"
        ),
        (
            "INDEXING_SEMANTIC_RELATION_MIN_CONFIDENCE="
            f"{values['INDEXING_SEMANTIC_RELATION_MIN_CONFIDENCE']}"
        ),
        (
            "INDEXING_SEMANTIC_RELATION_CONTEXT_CHARS="
            f"{values['INDEXING_SEMANTIC_RELATION_CONTEXT_CHARS']}"
        ),
        (
            "INDEXING_SEMANTIC_RELATION_MAX_PER_PAIR="
            f"{values['INDEXING_SEMANTIC_RELATION_MAX_PER_PAIR']}"
        ),
        (
            "INDEXING_SEMANTIC_RELATION_MODEL="
            f"{values['INDEXING_SEMANTIC_RELATION_MODEL']}"
        ),
        (
            "INDEXING_SEMANTIC_RELATION_TEMPERATURE="
            f"{values['INDEXING_SEMANTIC_RELATION_TEMPERATURE']}"
        ),
        "",

        # --------------------------------------------------------------------
        # Corpus Discovery
        # --------------------------------------------------------------------

        "# Corpus Discovery",
        (
            "INDEXING_DISCOVERY_LLM_ENABLED="
            f"{values['INDEXING_DISCOVERY_LLM_ENABLED']}"
        ),
        (
            "INDEXING_DISCOVERY_MAX_PAIRS="
            f"{values['INDEXING_DISCOVERY_MAX_PAIRS']}"
        ),
        (
            "INDEXING_DISCOVERY_BATCH_SIZE="
            f"{values['INDEXING_DISCOVERY_BATCH_SIZE']}"
        ),
        (
            "INDEXING_DISCOVERY_SAMPLE_CONTEXTS="
            f"{values['INDEXING_DISCOVERY_SAMPLE_CONTEXTS']}"
        ),
        (
            "INDEXING_DISCOVERY_MIN_EVIDENCE="
            f"{values['INDEXING_DISCOVERY_MIN_EVIDENCE']}"
        ),
        "",

        # --------------------------------------------------------------------
        # Relationship Discovery
        # --------------------------------------------------------------------

        "# Relationship Discovery",
        (
            "RELATIONSHIP_DISCOVERY_ENABLED="
            f"{values['RELATIONSHIP_DISCOVERY_ENABLED']}"
        ),
        (
            "RELATIONSHIP_DISCOVERY_MODE="
            f"{values['RELATIONSHIP_DISCOVERY_MODE']}"
        ),
        (
            "RELATIONSHIP_DISCOVERY_MAX_DOCUMENTS="
            f"{values['RELATIONSHIP_DISCOVERY_MAX_DOCUMENTS']}"
        ),
        (
            "RELATIONSHIP_DISCOVERY_MAX_CHUNKS_PER_DOCUMENT="
            f"{values['RELATIONSHIP_DISCOVERY_MAX_CHUNKS_PER_DOCUMENT']}"
        ),
        (
            "RELATIONSHIP_DISCOVERY_BATCH_SIZE="
            f"{values['RELATIONSHIP_DISCOVERY_BATCH_SIZE']}"
        ),
        (
            "RELATIONSHIP_DISCOVERY_MAX_CANDIDATES="
            f"{values['RELATIONSHIP_DISCOVERY_MAX_CANDIDATES']}"
        ),
        (
            "RELATIONSHIP_DISCOVERY_MIN_FREQUENCY="
            f"{values['RELATIONSHIP_DISCOVERY_MIN_FREQUENCY']}"
        ),
        (
            "RELATIONSHIP_DISCOVERY_CONTEXT_WINDOW="
            f"{values['RELATIONSHIP_DISCOVERY_CONTEXT_WINDOW']}"
        ),
        (
            "RELATIONSHIP_DISCOVERY_MAX_RELATIONSHIPS_PER_CHUNK="
            f"{values['RELATIONSHIP_DISCOVERY_MAX_RELATIONSHIPS_PER_CHUNK']}"
        ),
        (
            "RELATIONSHIP_DISCOVERY_USE_APPROVED_SCHEMA="
            f"{values['RELATIONSHIP_DISCOVERY_USE_APPROVED_SCHEMA']}"
        ),
        (
            "RELATIONSHIP_DISCOVERY_SCHEMA_PATH="
            f"{values['RELATIONSHIP_DISCOVERY_SCHEMA_PATH']}"
        ),
        "",

        # --------------------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------------------

        "# Retrieval",
        f"RETRIEVAL_TOP_K_VECTOR={values['RETRIEVAL_TOP_K_VECTOR']}",
        f"RETRIEVAL_TOP_K_FINAL={values['RETRIEVAL_TOP_K_FINAL']}",
        f"RETRIEVAL_VECTOR_THRESHOLD={values['RETRIEVAL_VECTOR_THRESHOLD']}",
        f"RETRIEVAL_MAX_HOPS={values['RETRIEVAL_MAX_HOPS']}",
        f"RETRIEVAL_PPR_ALPHA={values['RETRIEVAL_PPR_ALPHA']}",
        "",

        # --------------------------------------------------------------------
        # Agent
        # --------------------------------------------------------------------

        "# Agent",
        f"AGENT_MAX_RETRIES={values['AGENT_MAX_RETRIES']}",
        (
            "AGENT_RELEVANCE_THRESHOLD="
            f"{values['AGENT_RELEVANCE_THRESHOLD']}"
        ),
        "",
    ]

    return "\n".join(sections)


def save_runtime_settings(
    settings_dict: dict[str, Any],
) -> None:
    """Save runtime settings to runtime.env."""

    RUNTIME_ENV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for key, value in settings_dict.items():
        if value is not None:
            os.environ[key] = str(value)

    RUNTIME_ENV_PATH.write_text(
        _runtime_env_template(),
        encoding="utf-8",
    )

    logger.info(
        "Runtime settings saved to %s",
        RUNTIME_ENV_PATH,
    )


def clear_settings_cache() -> None:
    """Clear cached Pydantic settings."""

    from rag_core.config import get_settings

    if hasattr(get_settings, "cache_clear"):
        get_settings.cache_clear()


def reload_runtime_environment() -> None:
    """Reload runtime.env into environment."""

    if RUNTIME_ENV_PATH.exists():
        load_dotenv(
            dotenv_path=RUNTIME_ENV_PATH,
            override=True,
        )


def apply_runtime_settings() -> None:
    """Reload runtime.env and invalidate settings cache."""

    reload_runtime_environment()
    clear_settings_cache()


# ============================================================================
# Prompts configuration (config/prompts.json)
# ============================================================================

PROMPTS_FILE = APP_ROOT / "config" / "prompts.json"


def load_prompts() -> dict[str, Any]:
    """Load prompts from config/prompts.json."""
    if not PROMPTS_FILE.exists():
        return {"version": 1, "openai": {}}

    try:
        import json
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading prompts: %s", e)
        return {"version": 1, "openai": {}}


def save_prompts(prompts_dict: dict[str, Any]) -> None:
    """Save prompts to config/prompts.json."""
    PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        import json
        data = {
            "version": 1,
            "openai": prompts_dict
        }
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Prompts saved to %s", PROMPTS_FILE)
    except Exception as e:
        logger.error("Error saving prompts: %s", e)


def reload_prompts_cache() -> None:
    """Clear cached prompts in config."""
    from rag_core.config import _load_prompts
    
    # Force reload by calling the function (it doesn't cache internally)
    _load_prompts()


# Load runtime environment before importing get_settings.
load_runtime_environment()


from rag_core.config import get_settings


# ============================================================================
# Streamlit logging
# ============================================================================

class StreamlitLogHandler(logging.Handler):
    """Logging handler that stores logs in Streamlit session state."""

    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:
        try:
            log_entry = self.format(record)

            if "app_logs" not in st.session_state:
                st.session_state.app_logs = []

            if "app_log_signatures" not in st.session_state:
                st.session_state.app_log_signatures = set()

            signature = (
                record.levelname,
                record.name,
                record.getMessage(),
            )

            if signature in st.session_state.app_log_signatures:
                return

            st.session_state.app_log_signatures.add(
                signature
            )

            st.session_state.app_logs.append(
                {
                    "level": record.levelname,
                    "message": log_entry,
                    "name": record.name,
                }
            )

        except Exception:
            self.handleError(record)


def setup_streamlit_logging() -> None:
    """Configure application logging for Streamlit."""

    root_logger = logging.getLogger()

    for handler in list(root_logger.handlers):
        if isinstance(
            handler,
            StreamlitLogHandler,
        ):
            root_logger.removeHandler(handler)

    handler = StreamlitLogHandler()

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s: %(message)s",
            "%H:%M:%S",
        )
    )

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    for logger_name in (
        "rag_core",
        "agentic_graph_rag",
        "benchmark",
        "pymangle",
    ):
        module_logger = logging.getLogger(
            logger_name
        )

        module_logger.setLevel(logging.INFO)
        module_logger.propagate = True

        for handler in list(
            module_logger.handlers
        ):
            if isinstance(
                handler,
                StreamlitLogHandler,
            ):
                module_logger.removeHandler(
                    handler
                )


if "app_logs" not in st.session_state:
    st.session_state.app_logs = []

if "app_log_signatures" not in st.session_state:
    st.session_state.app_log_signatures = set()

setup_streamlit_logging()

logger = logging.getLogger(
    "rag_core.ui"
)


# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title="Agentic Graph RAG",
    page_icon="🔗",
    layout="wide",
)


# ============================================================================
# Sidebar
# ============================================================================

lang = st.sidebar.radio(
    "Language / Язык",
    ["en", "ru"],
    index=0,
)

t = get_translator(lang)

st.sidebar.title(
    t("app_title")
)

st.sidebar.caption(
    t("app_subtitle")
)

use_gpu = st.sidebar.checkbox(
    t("ingest_gpu"),
    value=False,
)

use_llm_router = st.sidebar.checkbox(
    "LLM Router"
    if lang == "en"
    else "LLM Роутер",
    value=False,
)

use_mangle_router = st.sidebar.checkbox(
    "Mangle Router"
    if lang == "en"
    else "Mangle Роутер",
    value=False,
)

API_URL = os.environ.get(
    "AGR_API_URL",
    "http://localhost:8507",
)


# ============================================================================
# Sidebar logs
# ============================================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📋 System Logs"
)

log_level_filter = st.sidebar.selectbox(
    "Filter Level",
    [
        "ALL",
        "INFO",
        "WARNING",
        "ERROR",
    ],
    index=0,
)

if st.sidebar.button("Clear Logs"):
    st.session_state.app_logs = []
    st.session_state.app_log_signatures = set()
    st.rerun()

with st.sidebar.expander(
    "Show Console Stream",
    expanded=False,
):
    filtered_logs = st.session_state.app_logs

    if log_level_filter != "ALL":
        filtered_logs = [
            log
            for log in filtered_logs
            if log["level"] == log_level_filter
        ]

    if filtered_logs:
        st.code(
            "\n".join(
                f"[{log['level']}] {log['message']}"
                for log in filtered_logs[-50:]
            ),
            language="log",
        )
    else:
        st.caption(
            "No logs recorded yet."
        )


# ============================================================================
# Cached resources
# ============================================================================

@st.cache_resource
def _get_neo4j_driver():
    from neo4j import GraphDatabase

    cfg = get_settings()

    driver = GraphDatabase.driver(
        cfg.neo4j.uri,
        auth=(
            cfg.neo4j.user,
            cfg.neo4j.password,
        ),
    )

    with driver.session() as session:
        session.run("RETURN 1").single()

    return driver


@st.cache_resource
def _get_openai_client():
    from rag_core.config import make_openai_client

    return make_openai_client(
        get_settings()
    )


@st.cache_resource
def _get_vector_store():
    from rag_core.vector_store import VectorStore

    store = VectorStore()
    store.init_index()

    return store


@st.cache_resource
def _get_reasoning_engine():
    from agentic_graph_rag.reasoning.reasoning_engine import (
        ReasoningEngine,
    )

    rules_dir = (
        APP_ROOT
        / "agentic_graph_rag"
        / "reasoning"
        / "rules"
    )

    return ReasoningEngine(
        str(rules_dir)
    )


@st.cache_resource
def _get_cache():
    try:
        from agentic_graph_rag.optimization.cache import (
            SubgraphCache,
        )

        return SubgraphCache()

    except Exception:

        class DummyCache:
            def stats(self):
                return {
                    "size": 0,
                    "max_size": 0,
                    "hit_rate": 0.0,
                }

        return DummyCache()


@st.cache_resource
def _get_monitor():
    try:
        from agentic_graph_rag.optimization.monitor import (
            QueryMonitor,
        )

        return QueryMonitor()

    except Exception:

        class DummyMonitor:
            def get_stats(self):
                return {
                    "total_queries": 0
                }

            def suggest_pagerank_weights(self):
                return {}

        return DummyMonitor()


# ============================================================================
# Session state
# ============================================================================

if "last_qa" not in st.session_state:
    st.session_state.last_qa = None

if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

if "discovery_candidates" not in st.session_state:
    st.session_state.discovery_candidates = []

if "discovery_relations" not in st.session_state:
    st.session_state.discovery_relations = []


# ============================================================================
# Tabs
# ============================================================================

(
    tab_ingest,
    tab_search,
    tab_graph,
    tab_discovery,
    tab_trace,
    tab_bench,
    tab_reasoning,
    tab_settings,
) = st.tabs(
    [
        t("tab_ingest"),
        t("tab_search"),
        t("tab_graph_explorer"),
        "Discovery",
        t("tab_agent_trace"),
        t("tab_benchmark"),
        t("tab_reasoning"),
        t("tab_settings"),
    ]
)


# ============================================================================
# TAB 1: INGEST
# ============================================================================

with tab_ingest:

    st.header(
        t("ingest_header")
    )

    st.caption(
        t("ingest_supported")
    )

    source = st.radio(
        t("ingest_upload"),
        [
            t("ingest_source_upload"),
            t("ingest_source_path"),
        ],
        horizontal=True,
    )

    uploaded_files = []
    file_paths: list[str] = []
    temp_files_created: list[str] = []

    if source == t(
        "ingest_source_upload"
    ):

        uploaded_files = st.file_uploader(
            t("ingest_upload"),
            type=[
                "txt",
                "pdf",
                "docx",
                "pptx",
                "xlsx",
                "html",
            ],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.success(
                f"Selected files: "
                f"{len(uploaded_files)}"
            )

        for uploaded in uploaded_files:
            st.caption(
                f"📄 {uploaded.name}"
            )

            suffix = Path(
                uploaded.name
            ).suffix

            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            )

            tmp.write(
                uploaded.read()
            )

            tmp.flush()
            tmp.close()

            file_paths.append(
                tmp.name
            )

            temp_files_created.append(
                tmp.name
            )

    else:

        file_path = st.text_input(
            t("ingest_path_input"),
            placeholder=t(
                "ingest_path_placeholder"
            ),
        )

        if (
            file_path
            and not Path(file_path).exists()
        ):
            st.warning(
                t(
                    "ingest_path_not_found",
                    path=file_path,
                )
            )

            file_path = None

        elif file_path:
            file_paths = [file_path]

    c1, c2 = st.columns(2)

    with c1:
        skip_enrichment = st.checkbox(
            t("ingest_skip_enrichment"),
            value=False,
        )

    with c2:
        build_graph = st.checkbox(
            "Build Knowledge Graph"
            if lang == "en"
            else "Построить граф знаний",
            value=True,
        )

    if st.button(
        t("ingest_button"),
        disabled=not file_paths,
        type="primary",
    ):

        apply_runtime_settings()

        with st.status(
            "Processing Document...",
            expanded=True,
        ) as status:

            try:

                cfg = get_settings()

                driver = _get_neo4j_driver()
                client = _get_openai_client()
                store = _get_vector_store()

                from rag_core.chunker import (
                    chunk_text,
                )

                from rag_core.embedder import (
                    embed_chunks,
                )

                from rag_core.enricher import (
                    enrich_chunks,
                )

                from rag_core.loader import (
                    load_file,
                )

                total_chunks_processed = 0

                for current_file in file_paths:

                    st.write(
                        f"📄 Loading: "
                        f"{current_file}"
                    )

                    text = load_file(
                        current_file,
                        use_gpu=use_gpu,
                    )
                    logger.warning(
                        "DISCOVERY DOCUMENT: %s | text_len=%d | preview=%r",
                        uploaded.name,
                        len(text or ""),
                        (text or "")[:500],
                    )

                    chunks = chunk_text(
                        text,
                        cfg.indexing.chunk_size,
                        cfg.indexing.chunk_overlap,
                    )

                    logger.warning(
                        "DISCOVERY CHUNKS: %s | chunks=%d",
                        uploaded.name,
                        len(chunks),
                    )

                    for i, chunk in enumerate(chunks[:3]):
                        logger.warning(
                            "DISCOVERY CHUNK DEBUG: index=%d id=%s text_len=%d preview=%r",
                            i,
                            chunk.id,
                            len(str(chunk.content or chunk.enriched_content or "")),
                            str(chunk.enriched_content or chunk.content or "")[:500],
                        )
                        
                    if not skip_enrichment:
                        chunks = enrich_chunks(
                            chunks,
                            text,
                        )

                    st.write(
                        "📐 Generating embeddings..."
                    )

                    chunks = embed_chunks(
                        chunks
                    )

                    total_chunks_processed += len(
                        chunks
                    )

                    store.add_chunks(
                        chunks
                    )

                    if build_graph:

                        st.write(
                            "🕸️ Building graph skeleton "
                            "+ semantic relations..."
                        )

                        from agentic_graph_rag.indexing.dual_node import (
                            build_dual_graph,
                            embed_phrase_nodes,
                            init_phrase_index,
                        )

                        from agentic_graph_rag.indexing.skeleton import (
                            build_skeleton_index,
                        )

                        embeddings = [
                            chunk.embedding
                            for chunk in chunks
                        ]

                        (
                            entities,
                            relationships,
                            skeletal,
                            peripheral,
                        ) = build_skeleton_index(
                            chunks,
                            embeddings,
                            openai_client=client,
                        )

                        (
                            phrase_nodes,
                            passage_nodes,
                            link_count,
                        ) = build_dual_graph(
                            entities,
                            chunks,
                            driver,
                            relationships=relationships,
                        )

                        if phrase_nodes:

                            embed_phrase_nodes(
                                phrase_nodes,
                                driver,
                                openai_client=client,
                            )

                            init_phrase_index(
                                driver
                            )

                        st.info(
                            f"Entities: "
                            f"{len(entities)} | "
                            f"Relations: "
                            f"{len(relationships)} | "
                            f"Semantic candidates "
                            f"were batched by "
                            f"{cfg.indexing.semantic_relation_batch_size}"
                        )

                        logger.info(
                            "Graph built: entities=%d, "
                            "relationships=%d, "
                            "skeletal=%d, "
                            "peripheral=%d",
                            len(entities),
                            len(relationships),
                            len(skeletal),
                            len(peripheral),
                        )

                status.update(
                    label=(
                        "Ingestion completed "
                        "successfully!"
                    ),
                    state="complete",
                    expanded=False,
                )

                st.success(
                    t(
                        "ingest_success",
                        chunks=total_chunks_processed,
                        total=store.count(),
                    )
                )

            except Exception as exc:

                logger.error(
                    "Ingestion failed: %s",
                    exc,
                    exc_info=True,
                )

                status.update(
                    label="Ingestion failed!",
                    state="error",
                )

                st.error(
                    t(
                        "error",
                        msg=str(exc),
                    )
                )

            finally:

                for temp_file in temp_files_created:

                    if os.path.exists(
                        temp_file
                    ):
                        os.unlink(
                            temp_file
                        )


# ============================================================================
# TAB 2: SEARCH
# ============================================================================

with tab_search:

    st.header(
        t("search_header")
    )

    mode = st.radio(
        t("search_mode"),
        [
            t("search_mode_vector"),
            t("search_mode_hybrid"),
            t("search_mode_agent"),
        ],
        horizontal=True,
    )

    query = st.text_input(
        t("search_input"),
        placeholder=t(
            "search_placeholder"
        ),
    )

    if st.button(
        t("search_button"),
        disabled=not query,
        type="primary",
    ):

        apply_runtime_settings()

        cfg = get_settings()

        mode_map = {
            t("search_mode_vector"): "vector",
            t("search_mode_hybrid"): "hybrid",
            t("search_mode_agent"): "agent_pattern",
        }

        api_mode = mode_map.get(
            mode,
            "agent_pattern",
        )

        if use_mangle_router:
            api_mode = "agent_mangle"

        elif use_llm_router:
            api_mode = "agent_llm"

        try:

            resp = httpx.post(
                f"{API_URL}/api/v1/query",
                json={
                    "text": query,
                    "mode": api_mode,
                },
                timeout=120.0,
            )

            resp.raise_for_status()

            from rag_core.models import (
                QAResult,
            )

            data = resp.json()

            qa = QAResult.model_validate(
                data
            )

            st.session_state.last_qa = qa

            st.session_state.last_trace = (
                data.get("trace")
            )

        except Exception as err:

            logger.warning(
                "API unavailable (%s). "
                "Using Direct Execution Mode.",
                err,
            )

            st.caption(
                "⚡ API unavailable — "
                "using Direct Execution Mode"
            )

            driver = _get_neo4j_driver()
            client = _get_openai_client()

            top_k = (
                cfg.retrieval.top_k_vector
            )

            if api_mode in (
                "agent_pattern",
                "agent_llm",
                "agent_mangle",
            ):

                from agentic_graph_rag.agent.retrieval_agent import (
                    run as agent_run,
                )

                reasoning = (
                    _get_reasoning_engine()
                    if use_mangle_router
                    else None
                )

                qa = agent_run(
                    query,
                    driver,
                    openai_client=client,
                    use_llm_router=use_llm_router,
                    reasoning=reasoning,
                )

            elif api_mode == "hybrid":

                from agentic_graph_rag.agent.tools import (
                    hybrid_search,
                )

                from rag_core.generator import (
                    generate_answer,
                )

                results = hybrid_search(
                    query,
                    driver,
                    client,
                    top_k=top_k,
                )

                qa = generate_answer(
                    query,
                    results,
                    client,
                )

            else:

                from agentic_graph_rag.agent.tools import (
                    vector_search,
                )

                from rag_core.generator import (
                    generate_answer,
                )

                results = vector_search(
                    query,
                    driver,
                    client,
                    top_k=top_k,
                )

                qa = generate_answer(
                    query,
                    results,
                    client,
                )

            st.session_state.last_qa = qa

            if qa.trace:

                st.session_state.last_trace = (
                    qa.trace.model_dump()
                )

            else:

                st.session_state.last_trace = {
                    "query": query,
                    "mode": api_mode,
                }

    qa = st.session_state.last_qa

    if qa:

        st.subheader(
            t("search_answer")
        )

        st.markdown(
            f">{qa.answer}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            t("search_confidence"),
            f"{qa.confidence:.0%}",
        )

        c2.metric(
            t(
                "search_retries",
                count=qa.retries,
            ),
            qa.retries,
        )

        if qa.router_decision:

            c3.metric(
                "Query Type",
                qa.router_decision.query_type.value,
            )

        if qa.sources:

            with st.expander(
                t(
                    "search_sources",
                    count=len(
                        qa.sources
                    ),
                )
            ):

                for i, src in enumerate(
                    qa.sources,
                    1,
                ):

                    st.markdown(
                        f"**{i}.** "
                        f"{src.chunk.content[:250]}..."
                    )

                    st.caption(
                        t(
                            "search_source_score",
                            score=src.score,
                        )
                    )


# ============================================================================
# TAB 3: GRAPH
# ============================================================================

with tab_graph:

    st.header(
        t("graph_header")
    )

    max_nodes = st.slider(
        t("graph_max_nodes"),
        10,
        300,
        80,
    )

    try:

        driver = _get_neo4j_driver()

        with driver.session() as session:

            phrase_count = session.run(
                """
                MATCH (n:PhraseNode)
                RETURN count(n) AS cnt
                """
            ).single()["cnt"]

            passage_count = session.run(
                """
                MATCH (n:PassageNode)
                RETURN count(n) AS cnt
                """
            ).single()["cnt"]

            c1, c2 = st.columns(2)

            c1.metric(
                t("graph_phrase_nodes"),
                phrase_count,
            )

            c2.metric(
                t("graph_passage_nodes"),
                passage_count,
            )

            result = session.run(
                """
                MATCH (a:PhraseNode)-[r]->(b)
                RETURN
                    a.name AS src,
                    type(r) AS rel,
                    coalesce(b.name, b.id, "?") AS tgt
                LIMIT $limit
                """,
                limit=max_nodes,
            )

            edges = [
                (
                    record["src"],
                    record["rel"],
                    record["tgt"],
                )
                for record in result
            ]

            if edges:

                dot_lines = [
                    "digraph G {",
                    "  rankdir=LR;",
                    (
                        '  node [shape=box, '
                        'style="rounded"];'
                    ),
                    '  edge [fontsize=9];',
                ]

                for src, rel, tgt in edges:

                    s_src = (
                        str(src or "?")
                        .replace('"', '\\"')
                    )

                    s_tgt = (
                        str(tgt or "?")
                        .replace('"', '\\"')
                    )

                    s_rel = (
                        str(rel or "")
                        .replace('"', '\\"')
                    )

                    dot_lines.append(
                        f'  "{s_src}" -> "{s_tgt}" '
                        f'[label="{s_rel}"];'
                    )

                dot_lines.append(
                    "}"
                )

                st.graphviz_chart(
                    "\n".join(dot_lines)
                )

            else:

                st.info(
                    t("graph_no_data")
                )

    except Exception as exc:

        logger.error(
            "Graph Explorer error: %s",
            exc,
            exc_info=True,
        )

        st.warning(
            t(
                "error",
                msg=str(exc),
            )
        )


# ============================================================================
# TAB 4: DISCOVERY
# ============================================================================

# ============================================================================
# TAB 4: DISCOVERY
# ============================================================================

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
# ============================================================================
# LOAD DISCOVERY RESULT FROM DB WITHOUT LLM
# ============================================================================

def _get_discovery_runs_from_neo4j():
    """
    Returns saved Discovery runs from Neo4j.

    Does not call any LLM.
    """

    driver = _get_neo4j_driver()

    with driver.session() as session:

        records = session.run(
            """
            MATCH (run:DiscoveryRun)

            RETURN
                run.run_id AS run_id,
                run.corpus_hash AS corpus_hash,
                run.mode AS mode,
                run.documents AS documents,
                run.min_frequency AS min_frequency,
                run.candidate_count AS candidate_count,
                run.created_at AS created_at

            ORDER BY run.created_at DESC
            """
        )

        return [
            record.data()
            for record in records
        ]


def _load_discovery_from_neo4j_without_llm(
    run_id,
):
    """
    Loads an already completed Discovery result
    from Neo4j.

    IMPORTANT:
    No document loading.
    No entity extraction.
    No embeddings.
    No LLM calls.
    No relationship discovery.

    Only Neo4j -> JSON -> Streamlit.
    """

    driver = _get_neo4j_driver()

    with driver.session() as session:

        record = session.run(
            """
            MATCH (run:DiscoveryRun {
                run_id: $run_id
            })

            RETURN
                run.run_id AS run_id,
                run.corpus_hash AS corpus_hash,
                run.result_json AS result_json

            LIMIT 1
            """,
            run_id=run_id,
        ).single()

    if record is None:

        return None

    result_json = record[
        "result_json"
    ]

    if not result_json:

        return None

    result = json.loads(
        result_json
    )

    # ------------------------------------------------------------
    # Restore Streamlit state
    # ------------------------------------------------------------

    st.session_state[
        "relationship_discovery_result"
    ] = result
    
    st.session_state[
        "relationship_discovery_run_id"
    ] = record[
        "run_id"
    ]

    st.session_state[
        "relationship_discovery_corpus_hash"
    ] = record[
        "corpus_hash"
    ]

    return result


# ============================================================================
# Discovery persistence helpers
# ============================================================================

# ============================================================================
# LOAD DISCOVERY RESULT FROM DB WITHOUT LLM
# ============================================================================

def _get_discovery_runs_from_neo4j():
    """
    Returns saved Discovery runs from Neo4j.

    Does not call any LLM.
    """

    driver = _get_neo4j_driver()

    with driver.session() as session:

        records = session.run(
            """
            MATCH (run:DiscoveryRun)

            RETURN
                run.run_id AS run_id,
                run.corpus_hash AS corpus_hash,
                run.mode AS mode,
                run.documents AS documents,
                run.min_frequency AS min_frequency,
                run.candidate_count AS candidate_count,
                run.created_at AS created_at

            ORDER BY run.created_at DESC
            """
        )

        return [
            record.data()
            for record in records
        ]


def _load_discovery_from_neo4j_without_llm(
    run_id,
):
    """
    Loads an already completed Discovery result
    from Neo4j.

    IMPORTANT:
    No document loading.
    No entity extraction.
    No embeddings.
    No LLM calls.
    No relationship discovery.

    Only Neo4j -> JSON -> Streamlit.
    """

    driver = _get_neo4j_driver()

    with driver.session() as session:

        record = session.run(
            """
            MATCH (run:DiscoveryRun {
                run_id: $run_id
            })

            RETURN
                run.run_id AS run_id,
                run.corpus_hash AS corpus_hash,
                run.result_json AS result_json

            LIMIT 1
            """,
            run_id=run_id,
        ).single()

    if record is None:

        return None

    result_json = record[
        "result_json"
    ]

    if not result_json:

        return None

    result = json.loads(
        result_json
    )

    # ------------------------------------------------------------
    # Restore Streamlit state
    # ------------------------------------------------------------

    st.session_state[
        "relationship_discovery_result"
    ] = result

    st.session_state[
        "relationship_discovery_run_id"
    ] = record[
        "run_id"
    ]

    st.session_state[
        "relationship_discovery_corpus_hash"
    ] = record[
        "corpus_hash"
    ]

    return result
def _make_discovery_corpus_hash(
    discovery_files,
    discovery_documents,
    discovery_mode,
    discovery_min_frequency,
    cfg,
):
    """
    Creates a stable hash for the current Discovery configuration
    and uploaded documents.

    If the same documents/configuration are used again, the hash
    allows us to identify the Discovery run.
    """

    hasher = hashlib.sha256()

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    config_payload = {
        "documents_limit": int(
            discovery_documents
        ),
        "mode": str(
            discovery_mode
        ),
        "min_frequency": int(
            discovery_min_frequency
        ),
        "chunk_size": int(
            cfg.indexing.chunk_size
        ),
        "chunk_overlap": int(
            cfg.indexing.chunk_overlap
        ),
    }

    hasher.update(
        json.dumps(
            config_payload,
            sort_keys=True,
            ensure_ascii=False,
        ).encode(
            "utf-8"
        )
    )

    # ------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------

    for uploaded in discovery_files[
        :discovery_documents
    ]:

        hasher.update(
            uploaded.name.encode(
                "utf-8"
            )
        )

        hasher.update(
            uploaded.getvalue()
        )

    return hasher.hexdigest()


def _make_discovery_run_id(
    corpus_hash,
):
    """
    Creates a human-readable Discovery run ID.
    """

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        f"discovery_"
        f"{timestamp}_"
        f"{corpus_hash[:12]}"
    )

def _make_json_serializable(obj):
    """
    Recursively converts Neo4j objects and other non-JSON
    serializable values into plain Python structures.
    """

    # ------------------------------------------------------------
    # Primitive JSON types
    # ------------------------------------------------------------

    if obj is None:
        return None

    if isinstance(
        obj,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return obj

    # ------------------------------------------------------------
    # Dict
    # ------------------------------------------------------------

    if isinstance(obj, dict):

        return {
            str(key): _make_json_serializable(value)
            for key, value in obj.items()
        }

    # ------------------------------------------------------------
    # Lists / tuples / sets
    # ------------------------------------------------------------

    if isinstance(
        obj,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            _make_json_serializable(value)
            for value in obj
        ]

    # ------------------------------------------------------------
    # Neo4j Relationship
    # ------------------------------------------------------------

    try:

        from neo4j.graph import Relationship

        if isinstance(
            obj,
            Relationship,
        ):

            return {
                "type": obj.type,
                "element_id": getattr(
                    obj,
                    "element_id",
                    None,
                ),
                "properties": {
                    str(key): _make_json_serializable(value)
                    for key, value in dict(
                        obj
                    ).items()
                },
                "start_node": getattr(
                    obj.start_node,
                    "element_id",
                    None,
                ),
                "end_node": getattr(
                    obj.end_node,
                    "element_id",
                    None,
                ),
            }

    except Exception:
        pass

    # ------------------------------------------------------------
    # Neo4j Node
    # ------------------------------------------------------------

    try:

        from neo4j.graph import Node

        if isinstance(
            obj,
            Node,
        ):

            return {
                "element_id": getattr(
                    obj,
                    "element_id",
                    None,
                ),
                "labels": list(
                    obj.labels
                ),
                "properties": {
                    str(key): _make_json_serializable(value)
                    for key, value in dict(
                        obj
                    ).items()
                },
            }

    except Exception:
        pass

    # ------------------------------------------------------------
    # Neo4j Path
    # ------------------------------------------------------------

    try:

        from neo4j.graph import Path

        if isinstance(
            obj,
            Path,
        ):

            return {
                "nodes": [
                    _make_json_serializable(node)
                    for node in obj.nodes
                ],
                "relationships": [
                    _make_json_serializable(rel)
                    for rel in obj.relationships
                ],
            }

    except Exception:
        pass

    # ------------------------------------------------------------
    # Neo4j Record
    # ------------------------------------------------------------

    try:

        from neo4j import Record

        if isinstance(
            obj,
            Record,
        ):

            return {
                str(key): _make_json_serializable(
                    value
                )
                for key, value in obj.items()
            }

    except Exception:
        pass

    # ------------------------------------------------------------
    # Objects with __dict__
    # ------------------------------------------------------------

    if hasattr(
        obj,
        "__dict__",
    ):

        return {
            str(key): _make_json_serializable(
                value
            )
            for key, value in vars(
                obj
            ).items()
            if not key.startswith("_")
        }

    # ------------------------------------------------------------
    # Final fallback
    # ------------------------------------------------------------

    return str(obj)

def _save_discovery_to_temp_file(
    result,
    run_id,
):
    """
    Saves Discovery result into a temporary JSON file.

    The path is stored in Streamlit session state so that
    subsequent reruns can access the file.
    """

    payload = {
        "run_id": run_id,
        "saved_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "result": result,
    }

    tmp_dir = tempfile.gettempdir()

    path = Path(
        tmp_dir
    ) / (
        f"{run_id}.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    st.session_state[
        "relationship_discovery_temp_file"
    ] = str(
        path
    )

    return path


def _load_discovery_from_temp_file():
    """
    Loads Discovery result from the temporary JSON file.
    """

    path = st.session_state.get(
        "relationship_discovery_temp_file"
    )

    if not path:
        return None

    path = Path(
        path
    )

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )

    result = payload.get(
        "result"
    )

    if result is None:
        return None

    st.session_state[
        "relationship_discovery_result"
    ] = result

    st.session_state[
        "relationship_discovery_run_id"
    ] = payload.get(
        "run_id"
    )

    return result


def _clear_discovery_temp_file():
    """
    Deletes the temporary Discovery result file.
    """

    path = st.session_state.get(
        "relationship_discovery_temp_file"
    )

    if not path:
        return False

    path = Path(
        path
    )

    deleted = False

    try:

        if path.exists():

            path.unlink()

            deleted = True

    except Exception as exc:

        logger.warning(
            "Failed to remove Discovery "
            "temporary file: %s",
            exc,
        )

    st.session_state.pop(
        "relationship_discovery_temp_file",
        None,
    )

    return deleted


def _save_discovery_to_neo4j(
    result,
    run_id,
    corpus_hash,
    discovery_mode,
    discovery_documents,
    discovery_min_frequency,
    cfg,
):
    """
    Persists the entire Discovery result into Neo4j.

    One Discovery execution = one :DiscoveryRun node.

    The complete result is stored as JSON in `result_json`.
    """

    driver = _get_neo4j_driver()

    candidates = result.get(
        "candidates",
        [],
    )

    payload = {
        "run_id": run_id,
        "corpus_hash": corpus_hash,
        "mode": discovery_mode,
        "documents": int(
            discovery_documents
        ),
        "min_frequency": int(
            discovery_min_frequency
        ),
        "chunk_size": int(
            cfg.indexing.chunk_size
        ),
        "chunk_overlap": int(
            cfg.indexing.chunk_overlap
        ),
        "candidate_count": len(
            candidates
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    result_json = json.dumps(
        result,
        ensure_ascii=False,
    )

    with driver.session() as session:

        session.run(
            """
            MERGE (run:DiscoveryRun {
                run_id: $run_id
            })

            SET
                run.corpus_hash = $corpus_hash,
                run.mode = $mode,
                run.documents = $documents,
                run.min_frequency = $min_frequency,
                run.chunk_size = $chunk_size,
                run.chunk_overlap = $chunk_overlap,
                run.candidate_count = $candidate_count,
                run.created_at = $created_at,
                run.result_json = $result_json

            RETURN run.run_id AS run_id
            """,
            run_id=payload[
                "run_id"
            ],
            corpus_hash=payload[
                "corpus_hash"
            ],
            mode=payload[
                "mode"
            ],
            documents=payload[
                "documents"
            ],
            min_frequency=payload[
                "min_frequency"
            ],
            chunk_size=payload[
                "chunk_size"
            ],
            chunk_overlap=payload[
                "chunk_overlap"
            ],
            candidate_count=payload[
                "candidate_count"
            ],
            created_at=payload[
                "created_at"
            ],
            result_json=result_json,
        ).consume()

    return payload


# ============================================================================
# SAVE SELECTED DISCOVERY RELATIONSHIPS TO GRAPH
# ============================================================================

def _normalize_relationship_type(value: str) -> str:
    """
    Converts a discovered relationship name into a safe Neo4j
    relationship type.

    Examples:
        "depends on" -> "DEPENDS_ON"
        "related-to" -> "RELATED_TO"
        "Uses"       -> "USES"
    """
    import re

    value = str(value or "").strip().upper()

    value = re.sub(
        r"[^A-Z0-9_]+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    ).strip("_")

    return value or "RELATED_TO"


def _extract_candidate_pairs(candidate):
    """
    Extracts source/target pairs from a Discovery candidate.

    Preferred format:

        candidate["pairs"] = [
            {
                "source": "...",
                "target": "...",
            }
        ]

    If full pairs are not available, falls back to the
    sample pairs stored in candidate["examples"].

    IMPORTANT:
    The current Discovery implementation stores examples,
    not necessarily all discovered pairs. Therefore the fallback
    can materialize only those examples.
    """

    pairs = candidate.get(
        "pairs",
        [],
    )

    normalized = []

    for pair in pairs:
        source = pair.get("source")
        target = pair.get("target")

        if source and target:
            normalized.append(
                {
                    "source": str(source).strip(),
                    "target": str(target).strip(),
                }
            )

    if normalized:
        return normalized

    # ------------------------------------------------------------
    # Fallback to examples
    # ------------------------------------------------------------

    examples = candidate.get(
        "examples",
        [],
    )

    for example in examples:
        source = example.get("source")
        target = example.get("target")

        if source and target:
            normalized.append(
                {
                    "source": str(source).strip(),
                    "target": str(target).strip(),
                }
            )

    return normalized


def _save_selected_relationships_without_llm(
    candidates,
    run_id,
):
    """
    Saves selected Discovery relationships directly into Neo4j.

    NO LLM is called.

    For each selected candidate:
        source entity
        target entity
        relationship type

    are materialized directly in the graph.

    If candidate["pairs"] exists, all those pairs are saved.

    Otherwise candidate["examples"] are used as a fallback.
    """

    driver = _get_neo4j_driver()

    saved_relationships = 0
    saved_types = 0

    with driver.session() as session:

        for candidate in candidates:

            relation = _normalize_relationship_type(
                candidate.get(
                    "relation",
                    "RELATED_TO",
                )
            )

            pairs = _extract_candidate_pairs(
                candidate
            )

            if not pairs:
                continue

            saved_types += 1

            for pair in pairs:

                source = pair["source"]
                target = pair["target"]

                query = f"""
                MERGE (source:Entity {{
                    name: $source
                }})

                MERGE (target:Entity {{
                    name: $target
                }})

                MERGE (source)-[r:{relation}]->(target)

                SET
                    r.discovery_run_id = $run_id,
                    r.discovery_frequency = $frequency,
                    r.discovery_unique_pairs = $unique_pairs

                RETURN type(r) AS relation
                """

                session.run(
                    query,
                    source=source,
                    target=target,
                    run_id=run_id,
                    frequency=int(
                        candidate.get(
                            "frequency",
                            0,
                        )
                    ),
                    unique_pairs=int(
                        candidate.get(
                            "unique_pairs",
                            0,
                        )
                    ),
                ).consume()

                saved_relationships += 1

    return {
        "relationship_types": saved_types,
        "relationships": saved_relationships,
    }


def _llm_validate_relationship(
    candidate,
    openai_client,
    model,
):
    """
    Uses LLM to validate a discovered relationship type.

    Returns:

        {
            "approved": bool,
            "relation": "...",
            "confidence": float,
            "reason": "..."
        }
    """

    relation = candidate.get(
        "relation",
        "RELATED_TO",
    )

    examples = candidate.get(
        "examples",
        [],
    )

    example_text = []

    for example in examples[:10]:

        source = example.get(
            "source",
            "",
        )

        target = example.get(
            "target",
            "",
        )

        description = example.get(
            "description",
            "",
        )

        example_text.append(
            f"- {source} -> {relation} -> {target}"
            + (
                f" | {description}"
                if description
                else ""
            )
        )

    examples_block = "\n".join(
        example_text
    )

    prompt = f"""
You are validating a semantic relationship discovered
in a document corpus.

Candidate relationship:
{relation}

Frequency:
{candidate.get("frequency", 0)}

Unique entity pairs:
{candidate.get("unique_pairs", 0)}

Examples:
{examples_block}

Determine whether this relationship type is meaningful
and useful for a knowledge graph.

Return ONLY valid JSON:

{{
  "approved": true,
  "relation": "RELATION_TYPE",
  "confidence": 0.0,
  "reason": "short explanation"
}}

Rules:

1. approved must be true or false.
2. confidence must be between 0 and 1.
3. relation must be uppercase snake_case.
4. Reject vague or meaningless relationships.
5. Prefer precise semantic relationship types.
"""

    response = openai_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You validate semantic knowledge graph "
                    "relationships. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        return {
            "approved": False,
            "relation": relation,
            "confidence": 0.0,
            "reason": "Empty LLM response",
        }

    try:
        parsed = json.loads(
            content
        )
    except json.JSONDecodeError:

        return {
            "approved": False,
            "relation": relation,
            "confidence": 0.0,
            "reason": "Invalid JSON returned by LLM",
        }

    return {
        "approved": bool(
            parsed.get(
                "approved",
                False,
            )
        ),
        "relation": _normalize_relationship_type(
            parsed.get(
                "relation",
                relation,
            )
        ),
        "confidence": float(
            parsed.get(
                "confidence",
                0,
            )
        ),
        "reason": str(
            parsed.get(
                "reason",
                "",
            )
        ),
    }


def _save_selected_relationships_with_llm(
    candidates,
    run_id,
    cfg,
):
    """
    Validates selected relationships with LLM and then
    materializes approved relationships in Neo4j.
    """

    client = _get_openai_client()

    model = getattr(
        cfg.indexing,
        "semantic_relation_model",
        None,
    )

    if not model:
        model = cfg.openai.synthesis_model

    validated = []
    approved_candidates = []

    # ------------------------------------------------------------
    # LLM validation
    # ------------------------------------------------------------

    for candidate in candidates:

        validation = _llm_validate_relationship(
            candidate=candidate,
            openai_client=client,
            model=model,
        )

        validated.append(
            {
                "candidate": candidate,
                "validation": validation,
            }
        )

        if (
            validation["approved"]
            and validation["confidence"] >= 0.7
        ):
            candidate_copy = dict(
                candidate
            )

            candidate_copy[
                "relation"
            ] = validation[
                "relation"
            ]

            candidate_copy[
                "llm_confidence"
            ] = validation[
                "confidence"
            ]

            candidate_copy[
                "llm_reason"
            ] = validation[
                "reason"
            ]

            approved_candidates.append(
                candidate_copy
            )

    # ------------------------------------------------------------
    # Materialize approved relationships
    # ------------------------------------------------------------

    graph_result = (
        _save_selected_relationships_without_llm(
            candidates=approved_candidates,
            run_id=run_id,
        )
    )

    return {
        "validated": validated,
        "approved": len(
            approved_candidates
        ),
        "graph": graph_result,
    }
def _load_latest_discovery_from_neo4j():
    """
    Loads the latest DiscoveryRun from Neo4j.

    No LLM call is performed.
    """

    driver = _get_neo4j_driver()

    with driver.session() as session:

        record = session.run(
            """
            MATCH (run:DiscoveryRun)

            RETURN
                run.run_id AS run_id,
                run.corpus_hash AS corpus_hash,
                run.mode AS mode,
                run.documents AS documents,
                run.min_frequency AS min_frequency,
                run.created_at AS created_at,
                run.result_json AS result_json

            ORDER BY run.created_at DESC

            LIMIT 1
            """
        ).single()

    if record is None:
        return None

    result_json = record[
        "result_json"
    ]

    if not result_json:
        return None

    result = json.loads(
        result_json
    )

    st.session_state[
        "relationship_discovery_result"
    ] = result

    st.session_state[
        "relationship_discovery_run_id"
    ] = record[
        "run_id"
    ]

    st.session_state[
        "relationship_discovery_corpus_hash"
    ] = record[
        "corpus_hash"
    ]

    return result


def _load_discovery_by_run_id(
    run_id,
):
    """
    Loads a particular DiscoveryRun from Neo4j.
    """

    driver = _get_neo4j_driver()

    with driver.session() as session:

        record = session.run(
            """
            MATCH (run:DiscoveryRun {
                run_id: $run_id
            })

            RETURN
                run.run_id AS run_id,
                run.corpus_hash AS corpus_hash,
                run.result_json AS result_json

            LIMIT 1
            """,
            run_id=run_id,
        ).single()

    if record is None:
        return None

    result_json = record[
        "result_json"
    ]

    if not result_json:
        return None

    result = json.loads(
        result_json
    )

    st.session_state[
        "relationship_discovery_result"
    ] = result

    st.session_state[
        "relationship_discovery_run_id"
    ] = record[
        "run_id"
    ]

    st.session_state[
        "relationship_discovery_corpus_hash"
    ] = record[
        "corpus_hash"
    ]

    return result


# ============================================================================
# Initialize Discovery state
# ============================================================================


if (
    "relationship_discovery_result"
    not in st.session_state
):

    st.session_state[
        "relationship_discovery_result"
    ] = None


if (
    "relationship_discovery_run_id"
    not in st.session_state
):

    st.session_state[
        "relationship_discovery_run_id"
    ] = None


if (
    "relationship_discovery_corpus_hash"
    not in st.session_state
):

    st.session_state[
        "relationship_discovery_corpus_hash"
    ] = None


if (
    "relationship_discovery_temp_file"
    not in st.session_state
):

    st.session_state[
        "relationship_discovery_temp_file"
    ] = None


# ============================================================================
# Discovery UI
# ============================================================================


with tab_discovery:

    st.header(
        "🔬 Semantic Relationship Discovery"
    )

    st.caption(
        "Анализирует документы и автоматически "
        "выявляет повторяющиеся семантические связи "
        "без заранее заданного словаря."
    )

    cfg = get_settings()

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    st.subheader(
        "Discovery Configuration"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        discovery_mode = st.selectbox(
            "Extraction Mode",
            [
                "both",
                "llm",
                "heuristic",
            ],
            index=[
                "both",
                "llm",
                "heuristic",
            ].index(
                cfg.relationship_discovery.mode
            ),
            key="discovery_mode",
        )

    with col2:

        discovery_documents = st.number_input(
            "Documents",
            min_value=1,
            max_value=1000,
            value=int(
                cfg.relationship_discovery.max_documents
            ),
            key="discovery_documents",
        )

    with col3:

        discovery_min_frequency = st.number_input(
            "Minimum Frequency",
            min_value=1,
            max_value=100,
            value=int(
                cfg.relationship_discovery.min_frequency
            ),
            key="discovery_min_frequency",
        )

    st.info(
        "💡 Для первого эксперимента рекомендуются "
        "10 документов, mode=both и minimum frequency=1."
    )

    # ------------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------------

    st.divider()

    st.subheader(
        "Documents for Discovery"
    )

    discovery_files = st.file_uploader(
        "Upload documents",
        type=[
            "txt",
            "pdf",
            "docx",
            "pptx",
            "xlsx",
            "html",
        ],
        accept_multiple_files=True,
        key="discovery_files",
    )

    if discovery_files:

        st.success(
            f"Selected documents: "
            f"{len(discovery_files)}"
        )

        for uploaded in discovery_files:

            st.caption(
                f"📄 {uploaded.name}"
            )



    # ============================================================================
    # DISCOVERY
    # ============================================================================

    st.divider()

    # ============================================================================
    # RUN DISCOVERY
    # ============================================================================

    if st.button(
            "🔍 Run Discovery",
            disabled=not discovery_files,
            type="primary",
            key="run_discovery",
    ):

        st.session_state[
            "relationship_discovery_result"
        ] = {
            "mode": discovery_mode,
            "raw_relationships": [],
            "candidates": [],
        }

        apply_runtime_settings()

        cfg = get_settings()

        try:

            from rag_core.loader import load_file
            from rag_core.chunker import chunk_text
            from rag_core.enricher import enrich_chunks

            from agentic_graph_rag.indexing.relationship_discovery import (
                discover_relationships,
            )

            with st.status(
                    "Running relationship discovery...",
                    expanded=True,
            ) as status:

                # ------------------------------------------------------------
                # Corpus hash
                # ------------------------------------------------------------

                corpus_hash = (
                    _make_discovery_corpus_hash(
                        discovery_files,
                        discovery_documents,
                        discovery_mode,
                        discovery_min_frequency,
                        cfg,
                    )
                )

                run_id = (
                    _make_discovery_run_id(
                        corpus_hash
                    )
                )

                st.session_state[
                    "relationship_discovery_corpus_hash"
                ] = corpus_hash

                st.session_state[
                    "relationship_discovery_run_id"
                ] = run_id

                st.write(
                    f"Discovery run: `{run_id}`"
                )

                # ------------------------------------------------------------
                # Load documents
                # ------------------------------------------------------------

                all_chunks = []

                for uploaded in discovery_files[
                                :discovery_documents
                                ]:

                    suffix = Path(
                        uploaded.name
                    ).suffix

                    with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=suffix,
                    ) as tmp:

                        tmp.write(
                            uploaded.getvalue()
                        )

                        tmp_path = tmp.name

                    try:

                        text = load_file(
                            tmp_path,
                            use_gpu=use_gpu,
                        )

                        chunks = chunk_text(
                            text,
                            cfg.indexing.chunk_size,
                            cfg.indexing.chunk_overlap,
                        )

                        if chunks:
                            chunks = enrich_chunks(
                                chunks,
                                text,
                            )

                        all_chunks.extend(
                            chunks
                        )

                    finally:

                        if os.path.exists(
                                tmp_path
                        ):
                            os.unlink(
                                tmp_path
                            )

                st.write(
                    f"Loaded {len(all_chunks)} chunks."
                )

                if not all_chunks:
                    st.warning(
                        "No text chunks were produced."
                    )

                    status.update(
                        label="Discovery stopped.",
                        state="error",
                    )

                    st.stop()

                # ------------------------------------------------------------
                # Entity extraction
                # ------------------------------------------------------------

                client = _get_openai_client()

                from agentic_graph_rag.indexing.skeleton import (
                    extract_entities_full,
                )

                st.write(
                    "Extracting entities..."
                )

                entities, _ = (
                    extract_entities_full(
                        all_chunks,
                        openai_client=client,
                    )
                )
                
                logger.warning(
                    "DISCOVERY DEBUG: chunks=%d entities=%d",
                    len(all_chunks),
                    len(entities),
                )

                for entity in entities[:20]:
                    logger.warning(
                        "DISCOVERY ENTITY: name=%r metadata=%r",
                        entity.name,
                        entity.metadata,
                    )
                    
                st.write(
                    f"Found {len(entities)} entities."
                )

                # ------------------------------------------------------------
                # Relationship discovery
                # ------------------------------------------------------------

                st.write(
                    "Discovering semantic relationships..."
                )

                cfg.relationship_discovery.enabled = True
                cfg.relationship_discovery.mode = discovery_mode
                cfg.relationship_discovery.max_documents = int(
                    discovery_documents
                )
                cfg.relationship_discovery.min_frequency = int(
                    discovery_min_frequency
                )

                save_runtime_settings(
                    {
                        "RELATIONSHIP_DISCOVERY_ENABLED": "true",
                        "RELATIONSHIP_DISCOVERY_MODE": discovery_mode,
                        "RELATIONSHIP_DISCOVERY_MAX_DOCUMENTS": str(
                            discovery_documents
                        ),
                        "RELATIONSHIP_DISCOVERY_MIN_FREQUENCY": str(
                            discovery_min_frequency
                        ),
                    }
                )
                reload_runtime_environment()
                clear_settings_cache()
                cfg = get_settings()

                result = (
                    discover_relationships(
                        all_chunks,
                        entities,
                        openai_client=client,
                    )
                )

                # ------------------------------------------------------------
                # Sort candidates by popularity
                # ------------------------------------------------------------

                candidates = sorted(
                    result.get(
                        "candidates",
                        [],
                    ),
                    key=lambda item: (
                        int(
                            item.get(
                                "frequency",
                                0,
                            )
                        ),
                        int(
                            item.get(
                                "unique_pairs",
                                0,
                            )
                        ),
                    ),
                    reverse=True,
                )

                result[
                    "candidates"
                ] = candidates

                # ------------------------------------------------------------
                # Save into Streamlit state
                # ------------------------------------------------------------

                st.session_state[
                    "relationship_discovery_result"
                ] = result
                
                # ------------------------------------------------------------
                # Save Discovery run to Neo4j
                # ------------------------------------------------------------

                _save_discovery_to_neo4j(
                    result=result,
                    run_id=run_id,
                    corpus_hash=corpus_hash,
                    discovery_mode=discovery_mode,
                    discovery_documents=discovery_documents,
                    discovery_min_frequency=discovery_min_frequency,
                    cfg=cfg,
                )
                # ------------------------------------------------------------
                # Automatically create temp JSON
                # ------------------------------------------------------------

                try:

                    _save_discovery_to_temp_file(
                        result,
                        run_id,
                    )

                except Exception as exc:

                    logger.warning(
                        "Could not create automatic "
                        "Discovery temp file: %s",
                        exc,
                    )

                status.update(
                    label="Discovery completed!",
                    state="complete",
                    expanded=False,
                )

            st.success(
                f"Found {len(candidates)} "
                f"candidate relationship types."
            )

            # Force rerun so the result UI appears immediately
            st.rerun()

        except Exception as exc:

            logger.error(
                "Relationship discovery failed: %s",
                exc,
                exc_info=True,
            )

            st.error(
                f"Discovery failed: {exc}"
            )

    # ============================================================================
    # DISCOVERY RESULT
    # ============================================================================

    result = st.session_state.get(
        "relationship_discovery_result"
    )

    if result:

        candidates = result.get(
            "candidates",
            [],
        )

        st.divider()

        st.subheader(
            "🔥 Top discovered relationships"
        )

        if not candidates:

            raw_count = len(
                result.get(
                    "raw_relationships",
                    [],
                )
            )

            threshold = int(
                result.get(
                    "threshold",
                    discovery_min_frequency,
                )
            )

            if raw_count:
                st.warning(
                    f"Discovery found {raw_count} relationship occurrences, "
                    f"but none reached frequency >= {threshold}."
                )

                st.info(
                    "Set Minimum Frequency = 1 to see all discovered "
                    "relationship types."
                )
            else:
                st.warning(
                    "Discovery found no valid relationships."
                )

        else:

            st.caption(
                "Связи отсортированы по частоте "
                "встречаемости в корпусе. "
                "Выбери те, которые действительно "
                "имеют смысл для графа."
            )

            # ------------------------------------------------------------
            # Number of relationships to display
            # ------------------------------------------------------------

            max_top = min(
                50,
                len(candidates),
            )

            top_n = st.slider(
                "Количество связей",
                min_value=1,
                max_value=max_top,
                value=min(
                    10,
                    max_top,
                ),
                key="discovery_top_n",
            )

            top_candidates = candidates[
                             :top_n
                             ]

            # ------------------------------------------------------------
            # Selected relationships
            # ------------------------------------------------------------

            selected_candidates = []

            for index, candidate in enumerate(
                    top_candidates
            ):

                relation = candidate.get(
                    "relation",
                    "RELATED_TO",
                )

                frequency = int(
                    candidate.get(
                        "frequency",
                        0,
                    )
                )

                unique_pairs = int(
                    candidate.get(
                        "unique_pairs",
                        0,
                    )
                )

                methods = candidate.get(
                    "methods",
                    {},
                )

                examples = candidate.get(
                    "examples",
                    [],
                )

                checked = st.checkbox(
                    (
                        f"**{relation}** — "
                        f"{frequency} occurrences · "
                        f"{unique_pairs} unique pairs"
                    ),
                    value=False,
                    key=(
                        "discovery_selected_"
                        f"{index}_"
                        f"{relation}"
                    ),
                )

                with st.expander(
                        f"Examples — {relation}",
                        expanded=False,
                ):

                    if methods:
                        st.write(
                            "Methods:",
                            methods,
                        )

                    if examples:

                        for example in examples:

                            source = example.get(
                                "source",
                                "—",
                            )

                            target = example.get(
                                "target",
                                "—",
                            )

                            description = example.get(
                                "description",
                                "",
                            )

                            st.markdown(
                                f"**{source}** "
                                f"→ `{relation}` → "
                                f"**{target}**"
                            )

                            if description:
                                st.caption(
                                    description
                                )

                    else:

                        st.caption(
                            "No examples available."
                        )

                if checked:
                    selected_candidates.append(
                        candidate
                    )

            # ------------------------------------------------------------
            # Selection summary
            # ------------------------------------------------------------

            st.divider()

            st.subheader(
                "💾 Save selected relationships"
            )

            st.write(
                f"Selected: "
                f"**{len(selected_candidates)}** "
                f"of **{len(top_candidates)}**"
            )

            if not selected_candidates:

                st.info(
                    "Select at least one relationship "
                    "to enable saving."
                )

            else:

                # ========================================================
                # THREE SAVE BUTTONS
                # ========================================================

                save_col1, save_col2, save_col3 = (
                    st.columns(3)
                )

                # --------------------------------------------------------
                # 1. Save temporary
                # --------------------------------------------------------

                with save_col1:

                    if st.button(
                            "📄 Save temporarily",
                            key="save_discovery_temp_result",
                            use_container_width=True,
                    ):

                        try:

                            run_id = (
                                    st.session_state.get(
                                        "relationship_discovery_run_id"
                                    )
                                    or _make_discovery_run_id(
                                "session"
                            )
                            )

                            path = (
                                _save_discovery_to_temp_file(
                                    result,
                                    run_id,
                                )
                            )

                            st.success(
                                f"Saved:\n{path}"
                            )

                        except Exception as exc:

                            logger.error(
                                "Failed to save "
                                "Discovery temp file",
                                exc_info=True,
                            )

                            st.error(
                                f"Failed to save: {exc}"
                            )

                # --------------------------------------------------------
                # 2. Save to Graph DB WITHOUT LLM
                # --------------------------------------------------------

                with save_col2:

                    if st.button(
                            "⚡ Save to Graph DB\nWITHOUT LLM",
                            key="save_discovery_graph_no_llm",
                            type="primary",
                            use_container_width=True,
                    ):

                        try:

                            run_id = (
                                    st.session_state.get(
                                        "relationship_discovery_run_id"
                                    )
                                    or "discovery_session"
                            )

                            graph_result = (
                                _save_selected_relationships_without_llm(
                                    candidates=selected_candidates,
                                    run_id=run_id,
                                )
                            )

                            st.success(
                                "✅ Saved directly to Graph DB"
                            )

                            st.json(
                                graph_result
                            )

                        except Exception as exc:

                            logger.error(
                                "Failed to save Discovery "
                                "relationships without LLM",
                                exc_info=True,
                            )

                            st.error(
                                "Failed to save to Graph DB "
                                f"without LLM: {exc}"
                            )

                # --------------------------------------------------------
                # 3. Validate + Save with LLM
                # --------------------------------------------------------

                with save_col3:

                    if st.button(
                            "🧠 Validate & Save\nWITH LLM",
                            key="save_discovery_graph_llm",
                            type="primary",
                            use_container_width=True,
                    ):

                        try:

                            cfg = get_settings()

                            run_id = (
                                    st.session_state.get(
                                        "relationship_discovery_run_id"
                                    )
                                    or "discovery_session"
                            )

                            with st.spinner(
                                    "LLM is validating relationships..."
                            ):

                                llm_result = (
                                    _save_selected_relationships_with_llm(
                                        candidates=selected_candidates,
                                        run_id=run_id,
                                        cfg=cfg,
                                    )
                                )

                            st.success(
                                "✅ LLM validation completed"
                            )

                            st.metric(
                                "Approved relationships",
                                llm_result[
                                    "approved"
                                ],
                            )

                            st.json(
                                llm_result[
                                    "graph"
                                ]
                            )

                            # ------------------------------------------------
                            # Show validation details
                            # ------------------------------------------------

                            with st.expander(
                                    "LLM validation details",
                                    expanded=False,
                            ):

                                for item in (
                                        llm_result[
                                            "validated"
                                        ]
                                ):
                                    candidate = item[
                                        "candidate"
                                    ]

                                    validation = item[
                                        "validation"
                                    ]

                                    st.write(
                                        candidate.get(
                                            "relation",
                                            "—",
                                        )
                                    )

                                    st.json(
                                        validation
                                    )

                        except Exception as exc:

                            logger.error(
                                "Failed to validate/save "
                                "Discovery relationships "
                                "with LLM",
                                exc_info=True,
                            )

                            st.error(
                                "Failed to save with LLM: "
                                f"{exc}"
                            )

    # ============================================================================
    # DISCOVERY HISTORY
    # ============================================================================

    st.divider()

    st.subheader(
        "📦 Saved Discovery Results"
    )

    try:

        discovery_runs = (
            _get_discovery_runs_from_neo4j()
        )

    except Exception as exc:

        discovery_runs = []

        st.warning(
            f"Could not load Discovery history: {exc}"
        )

    if discovery_runs:

        run_options = {
            (
                f"{run['run_id']} | "
                f"{run.get('mode', '—')} | "
                f"{run.get('documents', 0)} docs | "
                f"{run.get('candidate_count', 0)} candidates"
            ): run["run_id"]
            for run in discovery_runs
        }

        selected_run_label = st.selectbox(
            "Select saved Discovery run",
            list(
                run_options.keys()
            ),
            key="selected_discovery_run",
        )

        selected_run_id = run_options[
            selected_run_label
        ]

        if st.button(
                "📦 Load selected result WITHOUT LLM",
                key="load_discovery_without_llm",
                use_container_width=True,
        ):

            try:

                loaded = (
                    _load_discovery_from_neo4j_without_llm(
                        selected_run_id
                    )
                )

                if loaded:

                    st.success(
                        "✅ Discovery result loaded "
                        "from Neo4j without LLM."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Discovery result was not found "
                        "in Neo4j."
                    )

            except Exception as exc:

                logger.error(
                    "Failed to load Discovery result "
                    "without LLM",
                    exc_info=True,
                )

                st.error(
                    f"Failed to load Discovery result: {exc}"
                )

    else:

        st.info(
            "No saved Discovery results in Neo4j yet."
        )


# ============================================================================
# TAB 5: AGENT TRACE
# ============================================================================

with tab_trace:

    st.header(
        t("trace_header")
    )

    trace_data = (
        st.session_state.last_trace
    )

    if not trace_data:

        st.info(
            t("trace_no_data")
        )

    else:

        if trace_data.get(
            "router_step"
        ):

            st.subheader(
                t("trace_routing")
            )

            rs = trace_data[
                "router_step"
            ]

            decision = rs.get(
                "decision",
                {},
            )

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            c1.metric(
                "Method",
                rs.get(
                    "method",
                    "—",
                ),
            )

            c2.metric(
                t("trace_query_type"),
                decision.get(
                    "query_type",
                    "—",
                ),
            )

            c3.metric(
                t("trace_confidence"),
                f"{decision.get('confidence', 0):.0%}",
            )

            c4.metric(
                t("trace_tool"),
                decision.get(
                    "suggested_tool",
                    "—",
                ),
            )

        if trace_data.get(
            "tool_steps"
        ):

            st.divider()

            st.subheader(
                "Tool Execution Steps"
            )

            for i, step in enumerate(
                trace_data[
                    "tool_steps"
                ],
                1,
            ):

                with st.container(
                    border=True
                ):

                    c1, c2, c3, c4 = (
                        st.columns(4)
                    )

                    c1.metric(
                        f"Step {i}",
                        step.get(
                            "tool_name",
                            "—",
                        ),
                    )

                    c2.metric(
                        "Results",
                        step.get(
                            "results_count",
                            0,
                        ),
                    )

                    c3.metric(
                        "Relevance",
                        (
                            f"{step.get('relevance_score', 0):.1f}"
                            "/5.0"
                        ),
                    )

                    c4.metric(
                        "Duration",
                        (
                            f"{step.get('duration_ms', 0)}"
                            "ms"
                        ),
                    )

        with st.expander(
            "Raw Trace JSON"
        ):

            st.json(
                trace_data
            )


# ============================================================================
# TAB 6: BENCHMARK
# ============================================================================

with tab_bench:

    st.header(
        t("bench_header")
    )

    bench_modes = st.multiselect(
        t("bench_mode"),
        [
            "vector",
            "cypher",
            "hybrid",
            "agent_pattern",
            "agent_llm",
            "agent_mangle",
        ],
        default=[
            "vector",
            "hybrid",
            "agent_pattern",
        ],
    )

    if st.button(
        t("bench_run"),
        disabled=not bench_modes,
        type="primary",
    ):

        try:

            apply_runtime_settings()

            driver = (
                _get_neo4j_driver()
            )

            client = (
                _get_openai_client()
            )

            from benchmark.compare import (
                compare_modes,
                compute_metrics,
            )

            from benchmark.runner import (
                load_questions,
                run_benchmark,
            )

            questions = (
                load_questions()
            )

            all_results = (
                run_benchmark(
                    driver,
                    client,
                    modes=bench_modes,
                    questions=questions,
                    lang=lang,
                )
            )

            comparison = (
                compare_modes(
                    all_results
                )
            )

            st.dataframe(
                comparison,
                use_container_width=True,
            )

            for mode_name, results in (
                all_results.items()
            ):

                metrics = (
                    compute_metrics(
                        results
                    )
                )

                with st.expander(
                    f"{mode_name}: "
                    f"{metrics['correct']}/"
                    f"{metrics['total']} "
                    f"({metrics['accuracy']:.0%})"
                ):

                    st.dataframe(
                        results,
                        use_container_width=True,
                    )

        except Exception as exc:

            logger.error(
                "Benchmark failed",
                exc_info=True,
            )

            st.error(
                t(
                    "error",
                    msg=str(exc),
                )
            )


# ============================================================================
# TAB 7: REASONING
# ============================================================================


@st.cache_resource
def _get_reasoning_engine_from_sources(
    source_name: str,
    rules_text: str,
):

    from agentic_graph_rag.reasoning.reasoning_engine import (
        ReasoningEngine,
    )

    return (
        ReasoningEngine.from_sources(
            {
                source_name: rules_text,
            }
        )
    )


with tab_reasoning:

    st.header(
        t("reasoning_header")
    )

    default_rules_dir = (
        APP_ROOT
        / "agentic_graph_rag"
        / "reasoning"
        / "rules"
    )

    default_sources: dict[
        str,
        str,
    ] = {}

    if default_rules_dir.exists():

        for path in sorted(
            default_rules_dir.glob(
                "*.mg"
            )
        ):

            default_sources[
                path.stem
            ] = (
                path.read_text(
                    encoding="utf-8"
                )
            )

    source_names = (
        list(
            default_sources.keys()
        )
        if default_sources
        else ["routing"]
    )

    selected_source = st.selectbox(
        t("reasoning_rules_label"),
        source_names,
    )

    default_text = (
        default_sources.get(
            selected_source,
            "% Write Mangle rules here\n",
        )
    )

    rules_text = st.text_area(
        t("reasoning_rules_help"),
        value=default_text,
        height=250,
        key=f"rules_{selected_source}",
    )

    test_query = st.text_input(
        t("reasoning_query_label"),
        placeholder=t(
            "reasoning_query_placeholder"
        ),
        key="reasoning_query",
    )

    reasoning_engine = (
        _get_reasoning_engine_from_sources(
            selected_source,
            rules_text,
        )
    )

    if st.button(
        t("reasoning_run"),
        disabled=not test_query,
    ):

        try:

            result = (
                reasoning_engine.classify_query(
                    test_query
                )
            )

            if result:

                st.success(
                    "Route Matched! "
                    f"Tool: **{result['tool']}**"
                )

            else:

                st.warning(
                    t(
                        "reasoning_no_match"
                    )
                )

        except Exception as exc:

            st.error(
                t(
                    "reasoning_error",
                    msg=str(exc),
                )
            )

    try:

        strata = (
            reasoning_engine.get_strata(
                selected_source
            )
        )

        if strata:

            mermaid_lines = [
                "graph TD"
            ]

            for idx, predicates in enumerate(
                strata
            ):

                node_id = (
                    f"S{idx}"
                )

                label = (
                    f"Stratum {idx}: "
                    f"{', '.join(predicates[:3])}"
                )

                mermaid_lines.append(
                    f'    {node_id}["{label}"]'
                )

                if idx > 0:

                    mermaid_lines.append(
                        f"S{idx - 1} --> {node_id}"
                    )

            st.markdown(
                "```mermaid\n"
                + "\n".join(
                    mermaid_lines
                )
                + "\n```"
            )

    except Exception as exc:

        st.error(
            t(
                "reasoning_error",
                msg=str(exc),
            )
        )


# ============================================================================
# TAB 8: SETTINGS
# ============================================================================

@st.cache_data(ttl=3600)
def _get_embedding_dimensions(
    base_url: str,
    api_key: str,
    model: str,
) -> int | None:

    try:

        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "api_key": api_key or "none",
        }

        if base_url:
            kwargs["base_url"] = base_url

        client = OpenAI(
            **kwargs
        )

        response = client.embeddings.create(
            model=model,
            input="dimension probe",
        )

        if not response.data:
            return None

        return len(
            response.data[0].embedding
        )

    except Exception as exc:

        logger.warning(
            "Could not determine embedding "
            "dimensions for '%s': %s",
            model,
            exc,
        )

        return None


@st.cache_data(ttl=300)
def _fetch_available_models(
    base_url: str = "",
    api_key: str = "",
    for_embedding: bool = False,
) -> list[str]:

    try:
        from rag_core.config import make_openai_client

        client = make_openai_client(
            get_settings(),
            for_embedding=for_embedding,
        )

        if base_url:
            client = client.__class__(
                api_key=api_key or "none",
                base_url=base_url,
            )

        models_page = client.models.list()
        ids = [model.id for model in models_page.data]
        if ids:
            return sorted(ids)

    except Exception as exc:

        logger.warning(
            "Could not fetch model list for %s: %s",
            "embedding" if for_embedding else "general",
            exc,
        )

    cfg = get_settings()

    fallback = [
        cfg.openai.router_model,
        cfg.openai.cypher_model,
        cfg.openai.corrector_model,
        cfg.openai.synthesis_model,
        cfg.openai.embedding_model,
    ]

    if for_embedding:
        fallback = [
            cfg.openai.embedding_model,
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]

    return list(
        dict.fromkeys(
            [item for item in fallback if item]
        )
    )


def mask_api_key(
    api_key: str,
) -> str:

    if not api_key:
        return "Not set"

    if len(api_key) <= 8:
        return "********"

    return (
        api_key[:4]
        + "..."
        + api_key[-4:]
    )


with tab_settings:

    st.header(
        "Системные настройки и управление БД"
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    # Use one configuration object.
    # There is no need for a separate current_cfg.
    # ------------------------------------------------------------------------

    apply_runtime_settings()

    cfg = get_settings()

    available_models = (
        _fetch_available_models(
            base_url=cfg.openai.base_url,
            api_key=cfg.openai.api_key,
        )
    )

    # ========================================================================
    # OpenAI Connection
    # ========================================================================

    st.subheader(
        "OpenAI Connection"
    )

    st.caption("General LLM Endpoint (Router, Cypher, etc.)")
    col1, col2 = st.columns(2)

    with col1:

        openai_base_url = st.text_input(
            "General LLM Base URL",
            value=cfg.openai.base_url,
            key="settings_base_url",
        )

    with col2:

        openai_api_key = st.text_input(
            "General LLM API Key",
            value=cfg.openai.api_key,
            type="password",
            key="settings_api_key",
        )

    st.divider()
    st.caption("Embedding Endpoint (text-embedding-3-small, etc.) — Optional")
    
    col1, col2 = st.columns(2)

    with col1:

        embedding_base_url = st.text_input(
            "Embedding Base URL (leave empty to use General LLM endpoint)",
            value=cfg.openai.embedding_base_url,
            key="settings_embedding_base_url",
        )

    with col2:

        embedding_api_key = st.text_input(
            "Embedding API Key (leave empty to use General LLM API Key)",
            value=cfg.openai.embedding_api_key,
            type="password",
            key="settings_embedding_api_key",
        )

    if st.button(
        "Apply Provider",
        type="primary",
    ):

        save_runtime_settings(
            {
                "OPENAI_BASE_URL": (
                    openai_base_url
                ),
                "OPENAI_API_KEY": (
                    openai_api_key
                ),
                "OPENAI_EMBEDDING_BASE_URL": (
                    embedding_base_url
                ),
                "OPENAI_EMBEDDING_API_KEY": (
                    embedding_api_key
                ),
            }
        )

        reload_runtime_environment()
        clear_settings_cache()

        _fetch_available_models.clear()
        _get_embedding_dimensions.clear()

        try:
            st.cache_resource.clear()
        except Exception:
            pass

        st.success(
            "Provider settings applied."
        )

        st.rerun()

    st.divider()

    # ========================================================================
    # Embeddings
    # ========================================================================

    st.subheader(
        "Embeddings"
    )

    embedding_models = [
        model
        for model in _fetch_available_models(
            base_url=(
                cfg.openai.embedding_base_url
                or cfg.openai.base_url
            ),
            api_key=(
                cfg.openai.embedding_api_key
                or cfg.openai.api_key
            ),
            for_embedding=True,
        )
        if "embedding" in model.lower()
    ]

    if (
        cfg.openai.embedding_model
        and cfg.openai.embedding_model
        not in embedding_models
    ):

        embedding_models.insert(
            0,
            cfg.openai.embedding_model,
        )

    if embedding_models:

        if (
            cfg.openai.embedding_model
            in embedding_models
        ):

            embedding_index = (
                embedding_models.index(
                    cfg.openai.embedding_model
                )
            )

        else:

            embedding_index = 0

        embedding_model = st.selectbox(
            "Embedding Model",
            embedding_models,
            index=embedding_index,
            key="settings_embedding_model",
        )

        embedding_dimensions = (
            _get_embedding_dimensions(
                cfg.openai.embedding_base_url
                or cfg.openai.base_url,
                cfg.openai.embedding_api_key
                or cfg.openai.api_key,
                embedding_model,
            )
        )

        if embedding_dimensions is None:

            embedding_dimensions = (
                EMBEDDING_MODEL_DIMENSIONS.get(
                    embedding_model,
                    cfg.openai.embedding_dimensions,
                )
            )

        st.text_input(
            "Embedding Dimensions",
            value=str(
                embedding_dimensions
            ),
            disabled=True,
        )

        if st.button(
            "Apply Embedding Model"
        ):

            save_runtime_settings(
                {
                    "OPENAI_EMBEDDING_MODEL": (
                        embedding_model
                    ),
                    "OPENAI_EMBEDDING_DIMENSIONS": (
                        embedding_dimensions
                    ),
                }
            )

            reload_runtime_environment()
            clear_settings_cache()

            st.cache_resource.clear()

            st.success(
                "Embedding settings applied."
            )

            st.rerun()

    # ========================================================================
    # Main settings form
    # ========================================================================

    st.divider()

    with st.form(
        "settings_form"
    ):

        st.subheader(
            "Выбор моделей LLM"
        )

        def model_index(
            name: str,
        ) -> int:

            return (
                available_models.index(
                    name
                )
                if name in available_models
                else 0
            )

        col_m1, col_m2 = st.columns(2)

        with col_m1:

            router_model = st.selectbox(
                "Router Model",
                available_models,
                index=model_index(
                    cfg.openai.router_model
                ),
            )

            cypher_model = st.selectbox(
                "Cypher Generator Model",
                available_models,
                index=model_index(
                    cfg.openai.cypher_model
                ),
            )

        with col_m2:

            corrector_model = st.selectbox(
                "Cypher Corrector Model",
                available_models,
                index=model_index(
                    cfg.openai.corrector_model
                ),
            )

            synthesis_model = st.selectbox(
                "Answer Synthesis Model",
                available_models,
                index=model_index(
                    cfg.openai.synthesis_model
                ),
            )

        st.divider()

        # ====================================================================
        # LLM
        # ====================================================================

        st.subheader(
            "LLM Parameters"
        )

        llm_temperature = st.slider(
            "LLM Temperature",
            0.0,
            1.0,
            float(
                cfg.openai.llm_temperature
            ),
            step=0.1,
        )

        st.divider()

        # ====================================================================
        # LLM Prompts
        # ====================================================================

        st.subheader(
            "📝 System Prompts (LLM)"
        )

        router_system_prompt = st.text_area(
            "Router System Prompt",
            value=(
                cfg.openai.router_system_prompt
                if hasattr(cfg.openai, 'router_system_prompt')
                else "You are a routing agent. Analyze the query and route it appropriately."
            ),
            height=80,
            help="System prompt for the routing agent",
        )

        cypher_system_prompt = st.text_area(
            "Cypher Generator System Prompt",
            value=(
                cfg.openai.cypher_system_prompt
                if hasattr(cfg.openai, 'cypher_system_prompt')
                else "You are a Cypher query generator. Convert natural language to Neo4j Cypher."
            ),
            height=80,
            help="System prompt for Cypher query generation",
        )

        corrector_system_prompt = st.text_area(
            "Cypher Corrector System Prompt",
            value=(
                cfg.openai.corrector_system_prompt
                if hasattr(cfg.openai, 'corrector_system_prompt')
                else "You are a Cypher query corrector. Fix and optimize Cypher queries."
            ),
            height=80,
            help="System prompt for Cypher query correction",
        )

        synthesis_system_prompt = st.text_area(
            "Answer Synthesis System Prompt",
            value=(
                cfg.openai.synthesis_system_prompt
                if hasattr(cfg.openai, 'synthesis_system_prompt')
                else "You are a helpful assistant that synthesizes information into clear, accurate answers."
            ),
            height=80,
            help="System prompt for answer synthesis",
        )

        decomposer_system_prompt = st.text_area(
            "Query Decomposer System Prompt",
            value=(
                cfg.openai.decomposer_system_prompt
                if hasattr(cfg.openai, 'decomposer_system_prompt')
                else "You are a search query decomposer for a RAG system."
            ),
            height=80,
            help="System prompt for query decomposition into sub-queries",
        )

        entity_extraction_system_prompt = st.text_area(
            "Entity Extraction System Prompt",
            value=(
                cfg.openai.entity_extraction_system_prompt
                if hasattr(cfg.openai, 'entity_extraction_system_prompt')
                else "You are an entity extraction expert."
            ),
            height=80,
            help="System prompt for entity and relationship extraction",
        )

        relationship_discovery_system_prompt = st.text_area(
            "Relationship Discovery System Prompt",
            value=(
                cfg.openai.relationship_discovery_system_prompt
                if hasattr(cfg.openai, 'relationship_discovery_system_prompt')
                else "You are a precise knowledge graph relationship discovery system."
            ),
            height=80,
            help="System prompt for semantic relationship discovery",
        )

        st.divider()

        # ====================================================================
        # Indexing
        # ====================================================================

        st.subheader(
            "Индексация и чанкинг"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            chunk_size = st.number_input(
                "Chunk Size",
                100,
                4000,
                int(
                    cfg.indexing.chunk_size
                ),
                50,
            )

        with c2:

            chunk_overlap = st.number_input(
                "Chunk Overlap",
                0,
                1000,
                int(
                    cfg.indexing.chunk_overlap
                ),
                10,
            )

        with c3:

            skeleton_beta = st.number_input(
                "Skeleton Beta",
                0.0,
                1.0,
                float(
                    cfg.indexing.skeleton_beta
                ),
                0.05,
            )

        c4, c5 = st.columns(2)

        with c4:

            knn_k = st.number_input(
                "KNN K",
                1,
                1000,
                int(
                    cfg.indexing.knn_k
                ),
                1,
            )

        with c5:

            pagerank_damping = st.slider(
                "PageRank Damping",
                0.0,
                1.0,
                float(
                    cfg.indexing.pagerank_damping
                ),
                0.05,
            )

        st.divider()

        # ====================================================================
        # Semantic Relations
        # ====================================================================

        st.subheader(
            "🧬 Semantic Relations"
        )

        semantic_enabled = st.checkbox(
            "Enable semantic relations during ingest",
            value=bool(
                cfg.indexing.semantic_relations_enabled
            ),
        )

        semantic_relation_types = st.text_area(
            "Allowed relation types",
            value=(
                cfg.indexing.semantic_relation_types
            ),
            height=120,
        )

        c6, c7, c8 = st.columns(3)

        with c6:

            semantic_batch_size = st.number_input(
                "Relation Batch Size",
                1,
                100,
                int(
                    cfg.indexing.semantic_relation_batch_size
                ),
                1,
            )

        with c7:

            semantic_max_pairs = st.number_input(
                "Max Candidate Pairs",
                1,
                1000,
                int(
                    cfg.indexing.semantic_relation_max_pairs
                ),
                1,
            )

        with c8:

            semantic_min_cooccurrence = st.number_input(
                "Min Pair Evidence",
                1,
                100,
                int(
                    cfg.indexing.semantic_relation_min_cooccurrence
                ),
                1,
            )

        c9, c10, c11 = st.columns(3)

        with c9:

            semantic_min_confidence = st.slider(
                "Min Relation Confidence",
                0.0,
                1.0,
                float(
                    cfg.indexing.semantic_relation_min_confidence
                ),
                0.05,
            )

        with c10:

            semantic_max_per_pair = st.number_input(
                "Max Relations / Pair",
                1,
                20,
                int(
                    cfg.indexing.semantic_relation_max_per_pair
                ),
                1,
            )

        with c11:

            semantic_context_chars = st.number_input(
                "Context Chars",
                300,
                5000,
                int(
                    cfg.indexing.semantic_relation_context_chars
                ),
                100,
            )

        semantic_relation_model = st.text_input(
            "Semantic Relation Model "
            "(empty = Corrector Model)",
            value=(
                cfg.indexing.semantic_relation_model
            ),
        )

        semantic_relation_temperature = st.slider(
            "Semantic Relation Temperature",
            0.0,
            1.0,
            float(
                cfg.indexing.semantic_relation_temperature
            ),
            0.1,
        )

        st.divider()

        # ====================================================================
        # Corpus Discovery
        # ====================================================================

        st.subheader(
            "🔎 Corpus Discovery"
        )

        discovery_llm_enabled = st.checkbox(
            "Enable LLM in Discovery",
            value=bool(
                cfg.indexing.discovery_llm_enabled
            ),
        )

        c12, c13, c14 = st.columns(3)

        with c12:

            discovery_max_pairs = st.number_input(
                "Discovery Max Pairs",
                1,
                2000,
                int(
                    cfg.indexing.discovery_max_pairs
                ),
                1,
            )

        with c13:

            discovery_batch_size = st.number_input(
                "Discovery Batch Size",
                1,
                100,
                int(
                    cfg.indexing.discovery_batch_size
                ),
                1,
            )

        with c14:

            discovery_sample_contexts = st.number_input(
                "Contexts / Pair",
                1,
                10,
                int(
                    cfg.indexing.discovery_sample_contexts
                ),
                1,
            )

        discovery_min_evidence = st.number_input(
            "Discovery Min Evidence",
            1,
            100,
            int(
                cfg.indexing.discovery_min_evidence
            ),
            1,
        )

        st.divider()

        # ====================================================================
        # Retrieval
        # ====================================================================

        st.subheader(
            "Параметры поиска и графа"
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            top_k_vector = st.number_input(
                "Top-K Vector",
                1,
                100,
                int(
                    cfg.retrieval.top_k_vector
                ),
                1,
            )

            top_k_final = st.number_input(
                "Top-K Final",
                1,
                100,
                int(
                    cfg.retrieval.top_k_final
                ),
                1,
            )

        with r2:

            vector_threshold = st.slider(
                "Vector Threshold",
                0.0,
                1.0,
                float(
                    cfg.retrieval.vector_threshold
                ),
                0.05,
            )

            max_hops = st.number_input(
                "Max Hops",
                1,
                20,
                int(
                    cfg.retrieval.max_hops
                ),
                1,
            )

        with r3:

            ppr_alpha = st.slider(
                "PPR Alpha",
                0.0,
                1.0,
                float(
                    cfg.retrieval.ppr_alpha
                ),
                0.05,
            )

        st.divider()

        # ====================================================================
        # Agent
        # ====================================================================

        st.subheader(
            "Agent"
        )

        a1, a2 = st.columns(2)

        with a1:

            agent_max_retries = st.number_input(
                "Agent Max Retries",
                0,
                20,
                int(
                    cfg.agent.max_retries
                ),
                1,
            )

        with a2:

            agent_relevance_threshold = (
                st.number_input(
                    "Agent Relevance Threshold",
                    0.0,
                    10.0,
                    float(
                        cfg.agent.relevance_threshold
                    ),
                    0.1,
                )
            )

        submitted = st.form_submit_button(
            "Save & Apply Settings",
            type="primary",
        )

    # ========================================================================
    # Save settings
    # ========================================================================

    if submitted:

        # Save prompts to separate JSON file
        save_prompts(
            {
                "router_system_prompt": router_system_prompt,
                "cypher_system_prompt": cypher_system_prompt,
                "corrector_system_prompt": corrector_system_prompt,
                "synthesis_system_prompt": synthesis_system_prompt,
                "decomposer_system_prompt": decomposer_system_prompt,
                "entity_extraction_system_prompt": entity_extraction_system_prompt,
                "relationship_discovery_system_prompt": relationship_discovery_system_prompt,
            }
        )

        # Save other runtime settings
        save_runtime_settings(
            {
                # ------------------------------------------------------------
                # OpenAI
                # ------------------------------------------------------------

                "OPENAI_ROUTER_MODEL": (
                    router_model
                ),
                "OPENAI_CYPHER_MODEL": (
                    cypher_model
                ),
                "OPENAI_CORRECTOR_MODEL": (
                    corrector_model
                ),
                "OPENAI_SYNTHESIS_MODEL": (
                    synthesis_model
                ),
                "OPENAI_LLM_TEMPERATURE": (
                    llm_temperature
                ),

                # ------------------------------------------------------------
                # Indexing
                # ------------------------------------------------------------

                "INDEXING_CHUNK_SIZE": (
                    chunk_size
                ),
                "INDEXING_CHUNK_OVERLAP": (
                    chunk_overlap
                ),
                "INDEXING_SKELETON_BETA": (
                    skeleton_beta
                ),
                "INDEXING_KNN_K": (
                    knn_k
                ),
                "INDEXING_PAGERANK_DAMPING": (
                    pagerank_damping
                ),

                # ------------------------------------------------------------
                # Semantic Relations
                # ------------------------------------------------------------

                "INDEXING_SEMANTIC_RELATIONS_ENABLED": (
                    semantic_enabled
                ),
                "INDEXING_SEMANTIC_RELATION_TYPES": (
                    semantic_relation_types
                ),
                "INDEXING_SEMANTIC_RELATION_BATCH_SIZE": (
                    semantic_batch_size
                ),
                "INDEXING_SEMANTIC_RELATION_MAX_PAIRS": (
                    semantic_max_pairs
                ),
                "INDEXING_SEMANTIC_RELATION_MIN_COOCCURRENCE": (
                    semantic_min_cooccurrence
                ),
                "INDEXING_SEMANTIC_RELATION_MIN_CONFIDENCE": (
                    semantic_min_confidence
                ),
                "INDEXING_SEMANTIC_RELATION_CONTEXT_CHARS": (
                    semantic_context_chars
                ),
                "INDEXING_SEMANTIC_RELATION_MAX_PER_PAIR": (
                    semantic_max_per_pair
                ),
                "INDEXING_SEMANTIC_RELATION_MODEL": (
                    semantic_relation_model
                ),
                "INDEXING_SEMANTIC_RELATION_TEMPERATURE": (
                    semantic_relation_temperature
                ),

                # ------------------------------------------------------------
                # Corpus Discovery
                # ------------------------------------------------------------

                "INDEXING_DISCOVERY_LLM_ENABLED": (
                    discovery_llm_enabled
                ),
                "INDEXING_DISCOVERY_MAX_PAIRS": (
                    discovery_max_pairs
                ),
                "INDEXING_DISCOVERY_BATCH_SIZE": (
                    discovery_batch_size
                ),
                "INDEXING_DISCOVERY_SAMPLE_CONTEXTS": (
                    discovery_sample_contexts
                ),
                "INDEXING_DISCOVERY_MIN_EVIDENCE": (
                    discovery_min_evidence
                ),

                # ------------------------------------------------------------
                # Retrieval
                # ------------------------------------------------------------

                "RETRIEVAL_TOP_K_VECTOR": (
                    top_k_vector
                ),
                "RETRIEVAL_TOP_K_FINAL": (
                    top_k_final
                ),
                "RETRIEVAL_VECTOR_THRESHOLD": (
                    vector_threshold
                ),
                "RETRIEVAL_MAX_HOPS": (
                    max_hops
                ),
                "RETRIEVAL_PPR_ALPHA": (
                    ppr_alpha
                ),

                # ------------------------------------------------------------
                # Agent
                # ------------------------------------------------------------

                "AGENT_MAX_RETRIES": (
                    agent_max_retries
                ),
                "AGENT_RELEVANCE_THRESHOLD": (
                    agent_relevance_threshold
                ),
            }
        )

        reload_runtime_environment()
        clear_settings_cache()
        reload_prompts_cache()

        try:

            st.cache_resource.clear()
            st.cache_data.clear()

        except Exception:
            pass

        st.success(
            "Все настройки сохранены "
            "(config/prompts.json и runtime.env) и применены."
        )

        st.rerun()

    # ========================================================================
    # Current Settings
    # ========================================================================

    st.divider()

    st.subheader(
        "Current Settings"
    )

    # IMPORTANT:
    # Do not create current_cfg.
    # cfg is already the current Settings instance.

    cfg = get_settings()

    st.json(
        {
            # =================================================================
            # Neo4j
            # =================================================================

            "Neo4j URI": (
                cfg.neo4j.uri
            ),

            "Neo4j User": (
                cfg.neo4j.user
            ),

            "Neo4j Password": (
                mask_api_key(
                    cfg.neo4j.password
                )
            ),

            # =================================================================
            # OpenAI
            # =================================================================

            "OpenAI Base URL": (
                cfg.openai.base_url
            ),

            "OpenAI API Key": (
                mask_api_key(
                    cfg.openai.api_key
                )
            ),

            "Router Model": (
                cfg.openai.router_model
            ),

            "Cypher Generator Model": (
                cfg.openai.cypher_model
            ),

            "Cypher Corrector Model": (
                cfg.openai.corrector_model
            ),

            "Answer Synthesis Model": (
                cfg.openai.synthesis_model
            ),

            "Embedding Model": (
                cfg.openai.embedding_model
            ),

            "Embedding Dimensions": (
                cfg.openai.embedding_dimensions
            ),

            "LLM Temperature": (
                cfg.openai.llm_temperature
            ),

            # =================================================================
            # Indexing
            # =================================================================

            "Chunk Size": (
                cfg.indexing.chunk_size
            ),

            "Chunk Overlap": (
                cfg.indexing.chunk_overlap
            ),

            "Skeleton Beta": (
                cfg.indexing.skeleton_beta
            ),

            "KNN K": (
                cfg.indexing.knn_k
            ),

            "PageRank Damping": (
                cfg.indexing.pagerank_damping
            ),

            # =================================================================
            # Semantic Relations
            # =================================================================

            "Semantic Relations Enabled": (
                cfg.indexing.semantic_relations_enabled
            ),

            "Semantic Relation Types": (
                cfg.indexing.semantic_relation_types
            ),

            "Semantic Relation Batch Size": (
                cfg.indexing.semantic_relation_batch_size
            ),

            "Semantic Relation Max Pairs": (
                cfg.indexing.semantic_relation_max_pairs
            ),

            "Semantic Relation Min Cooccurrence": (
                cfg.indexing.semantic_relation_min_cooccurrence
            ),

            "Semantic Relation Min Confidence": (
                cfg.indexing.semantic_relation_min_confidence
            ),

            "Semantic Relation Context Chars": (
                cfg.indexing.semantic_relation_context_chars
            ),

            "Semantic Relation Max / Pair": (
                cfg.indexing.semantic_relation_max_per_pair
            ),

            "Semantic Relation Model": (
                cfg.indexing.semantic_relation_model
            ),

            "Semantic Relation Temperature": (
                cfg.indexing.semantic_relation_temperature
            ),

            # =================================================================
            # Corpus Discovery
            # =================================================================

            "Discovery LLM Enabled": (
                cfg.indexing.discovery_llm_enabled
            ),

            "Discovery Max Pairs": (
                cfg.indexing.discovery_max_pairs
            ),

            "Discovery Batch Size": (
                cfg.indexing.discovery_batch_size
            ),

            "Discovery Sample Contexts": (
                cfg.indexing.discovery_sample_contexts
            ),

            "Discovery Min Evidence": (
                cfg.indexing.discovery_min_evidence
            ),

            # =================================================================
            # Relationship Discovery
            # =================================================================

            "Relationship Discovery Enabled": (
                cfg.relationship_discovery.enabled
            ),

            "Relationship Discovery Mode": (
                cfg.relationship_discovery.mode
            ),

            "Relationship Discovery Max Documents": (
                cfg.relationship_discovery.max_documents
            ),

            "Relationship Discovery Max Chunks / Document": (
                cfg.relationship_discovery.max_chunks_per_document
            ),

            "Relationship Discovery Batch Size": (
                cfg.relationship_discovery.batch_size
            ),

            "Relationship Discovery Max Candidates": (
                cfg.relationship_discovery.max_candidates
            ),

            "Relationship Discovery Min Frequency": (
                cfg.relationship_discovery.min_frequency
            ),

            "Relationship Discovery Context Window": (
                cfg.relationship_discovery.context_window
            ),

            "Relationship Discovery Max Relations / Chunk": (
                cfg.relationship_discovery.max_relationships_per_chunk
            ),

            "Relationship Discovery Use Approved Schema": (
                cfg.relationship_discovery.use_approved_schema
            ),

            "Relationship Discovery Schema Path": (
                cfg.relationship_discovery.schema_path
            ),

            # =================================================================
            # Retrieval
            # =================================================================

            "Top K Vector": (
                cfg.retrieval.top_k_vector
            ),

            "Top K Final": (
                cfg.retrieval.top_k_final
            ),

            "Vector Threshold": (
                cfg.retrieval.vector_threshold
            ),

            "Max Hops": (
                cfg.retrieval.max_hops
            ),

            "PPR Alpha": (
                cfg.retrieval.ppr_alpha
            ),

            # =================================================================
            # Agent
            # =================================================================

            "Agent Max Retries": (
                cfg.agent.max_retries
            ),

            "Relevance Threshold": (
                cfg.agent.relevance_threshold
            ),
        }
    )

    # ========================================================================
    # Runtime Environment
    # ========================================================================

    st.divider()

    st.subheader(
        "Runtime Environment"
    )

    st.code(
        str(RUNTIME_ENV_PATH)
    )

    if RUNTIME_ENV_PATH.exists():

        st.success(
            "runtime.env найден."
        )

    else:

        st.warning(
            "runtime.env пока не существует."
        )

    # ========================================================================
    # Vector Store Statistics
    # ========================================================================

    st.subheader(
        "Vector Store Statistics"
    )

    try:

        store = _get_vector_store()

        st.write(
            f"Total Chunks: {store.count()}"
        )

    except Exception as exc:

        st.error(
            f"Error: {exc}"
        )

    # ========================================================================
    # Cache Statistics
    # ========================================================================

    st.subheader(
        "Cache Statistics"
    )

    try:

        cache_stats = (
            _get_cache().stats()
        )

        st.json(
            cache_stats
        )

    except Exception as exc:

        st.error(
            f"Error: {exc}"
        )

    # ========================================================================
    # Monitor Statistics
    # ========================================================================

    st.subheader(
        "Monitor Statistics"
    )

    try:

        monitor_stats = (
            _get_monitor().get_stats()
        )

        st.json(
            monitor_stats
        )

    except Exception as exc:

        st.error(
            f"Error: {exc}"
        )

    # ========================================================================
    # Clear Vector Store
    # ========================================================================

    st.divider()

    st.subheader(
        "Clear Vector Store"
    )

    confirm = st.text_input(
        "Type DELETE to clear the vector store:",
        key="clear_confirm",
    )

    if st.button(
        "Clear Vector Store",
        disabled=confirm != "DELETE",
    ):

        try:

            store = _get_vector_store()

            count = store.count()

            if hasattr(
                store,
                "delete_all",
            ):

                store.delete_all()

            elif hasattr(
                store,
                "clear",
            ):

                store.clear()

            else:

                raise AttributeError(
                    "Vector Store does not support "
                    "delete_all() or clear()."
                )

            st.success(
                "Vector Store cleared. "
                f"Removed {count} chunks."
            )

            st.cache_data.clear()
            st.cache_resource.clear()

            st.rerun()

        except Exception as exc:

            st.error(
                f"Error: {exc}"
            )

    # ========================================================================
    # Reset Graph & All Data
    # ========================================================================

    st.divider()

    st.subheader(
        "Reset Graph & All Data"
    )

    confirm_full = st.text_input(
        "Type DELETE ALL to clear Neo4j and Vector Store:",
        key="clear_full_confirm",
    )

    if st.button(
        "Clear Graph & Vectors",
        disabled=confirm_full != "DELETE ALL",
        type="primary",
    ):

        try:

            driver = _get_neo4j_driver()

            with driver.session() as session:

                session.run(
                    "MATCH (n) DETACH DELETE n"
                )

            store = _get_vector_store()

            count = store.count()

            if hasattr(
                store,
                "delete_all",
            ):

                store.delete_all()

            elif hasattr(
                store,
                "clear",
            ):

                store.clear()

            else:

                raise AttributeError(
                    "Vector Store does not support "
                    "delete_all() or clear()."
                )

            st.success(
                "Fully cleared. "
                f"Removed {count} vector chunks "
                "and all Neo4j graph nodes."
            )

            st.cache_data.clear()
            st.cache_resource.clear()

            st.rerun()

        except Exception as exc:

            st.error(
                f"Error: {exc}"
            )