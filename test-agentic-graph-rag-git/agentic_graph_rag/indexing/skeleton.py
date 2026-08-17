"""KET-RAG skeleton indexing + targeted semantic relation discovery.

Pipeline
--------
1. Build KNN graph over chunk embeddings.
2. Compute PageRank.
3. Select top-beta skeletal chunks.
4. Extract Entity objects from skeletal chunks.
5. Link peripheral chunks cheaply with MENTIONED_IN.
6. Generate candidate pairs ONLY between already extracted entities.
7. Rank candidates by corpus evidence / co-occurrence.
8. Optionally extract MULTIPLE semantic relations between each pair.
9. LLM may work with:
       a) an approved relation allow-list, OR
       b) dynamically discovered relation names.
10. Deduplicate relations and preserve evidence/confidence metadata.

Corpus Discovery
----------------
Discovery is intentionally separated from ingestion.

It can:
    - inspect the existing PhraseNode -> PassageNode graph;
    - find entity pairs repeatedly occurring in the same passages;
    - rank those pairs by evidence_count;
    - optionally send only the most important pairs to the LLM;
    - extract multiple semantic relations;
    - return a report before anything is written to Neo4j;
    - optionally apply approved relations to Neo4j.

This makes the graph construction cheap and universal.

Example
-------
For documents containing:

    Liver
    abdominal cavity
    portal vein
    blood circulation
    metabolism

the system can discover:

    Liver -> LOCATED_IN -> Abdominal_Cavity
    Liver -> CONNECTED_TO -> Portal_Vein
    Liver -> PARTICIPATES_IN -> Blood_Circulation
    Liver -> PARTICIPATES_IN -> Metabolism

No medical relation names are hard-coded in this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from rag_core.config import get_settings
from rag_core.models import Chunk, Entity, Relationship

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS FOR OPTIONAL CONFIGURATION
# ============================================================================


def _cfg_value(name: str, default: Any) -> Any:
    """Read an optional indexing setting safely.

    This keeps the skeleton backward-compatible while new settings are added
    to rag_core.config.IndexingSettings.
    """
    cfg = get_settings().indexing
    return getattr(cfg, name, default)


# ============================================================================
# 1. KNN GRAPH
# ============================================================================


def build_knn_graph(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    k: int | None = None,
) -> nx.DiGraph:
    """Build a directed KNN graph over chunk embeddings."""

    if k is None:
        k = get_settings().indexing.knn_k

    n = len(chunks)

    if n == 0:
        return nx.DiGraph()

    emb_matrix = np.asarray(
        embeddings,
        dtype=float,
    )

    norms = np.linalg.norm(
        emb_matrix,
        axis=1,
        keepdims=True,
    )

    norms = np.where(
        norms == 0,
        1.0,
        norms,
    )

    normed = emb_matrix / norms

    sim_matrix = normed @ normed.T

    graph = nx.DiGraph()

    for i in range(n):
        graph.add_node(
            i,
            chunk_id=chunks[i].id,
        )

    effective_k = min(
        max(k, 0),
        n - 1,
    )

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0

        if effective_k <= 0:
            continue

        top_indices = np.argsort(sims)[
            -effective_k:
        ][::-1]

        for j_idx in top_indices:
            j = int(j_idx)

            graph.add_edge(
                i,
                j,
                weight=float(
                    sims[j]
                ),
            )

    logger.info(
        "Built KNN graph: %d nodes, %d edges (k=%d)",
        n,
        graph.number_of_edges(),
        effective_k,
    )

    return graph


# ============================================================================
# 2. PAGERANK
# ============================================================================


def compute_pagerank(
    knn_graph: nx.DiGraph,
    damping: float | None = None,
) -> dict[int, float]:
    """Compute PageRank scores for chunk importance."""

    if damping is None:
        damping = (
            get_settings()
            .indexing
            .pagerank_damping
        )

    if knn_graph.number_of_nodes() == 0:
        return {}

    return nx.pagerank(
        knn_graph,
        alpha=damping,
        weight="weight",
    )


# ============================================================================
# 3. SKELETAL SELECTION
# ============================================================================


def select_skeletal_chunks(
    chunks: list[Chunk],
    pagerank_scores: dict[int, float],
    beta: float | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Split chunks into expensive skeletal and cheap peripheral chunks."""

    if beta is None:
        beta = (
            get_settings()
            .indexing
            .skeleton_beta
        )

    if not chunks or not pagerank_scores:
        return [], list(chunks)

    ranked = sorted(
        pagerank_scores.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    n_skeletal = max(
        1,
        min(
            len(chunks),
            int(
                len(chunks) * beta
            ),
        ),
    )

    skeletal_indices = {
        idx
        for idx, _ in ranked[
            :n_skeletal
        ]
    }

    skeletal = [
        chunk
        for i, chunk in enumerate(chunks)
        if i in skeletal_indices
    ]

    peripheral = [
        chunk
        for i, chunk in enumerate(chunks)
        if i not in skeletal_indices
    ]

    logger.info(
        "Selected %d skeletal + %d peripheral chunks (beta=%.2f)",
        len(skeletal),
        len(peripheral),
        beta,
    )

    return skeletal, peripheral


# ============================================================================
# 4. ENTITY EXTRACTION
# ============================================================================

def extract_entities_full(
    skeletal_chunks: list[Chunk],
    openai_client: OpenAI | None = None,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Extract entities and explicit relationships from skeletal chunks.

    IMPORTANT:
    This version intentionally uses a strict extraction prompt and a robust
    parser because the downstream pipeline depends on entities being detected.
    """

    if not skeletal_chunks:
        logger.warning(
            "Entity extraction skipped: skeletal_chunks is empty"
        )
        return [], []

    cfg = get_settings()

    if openai_client is None:
        from rag_core.config import make_openai_client

        openai_client = make_openai_client(cfg)

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Keep the extraction format explicit and stable.
    # Do NOT depend on an external prompt until entity extraction works.
    # ------------------------------------------------------------------

    system_prompt = """
You are an entity extraction engine for a knowledge graph.

Extract ALL important entities from the provided text.

Entities may include:
- people
- organizations
- companies
- products
- technologies
- software
- hardware
- concepts
- processes
- methods
- documents
- projects
- locations
- dates
- events
- technical terms
- domain-specific concepts

Do NOT extract ordinary stop words or generic words such as:
"system", "thing", "information", "example", "text"
unless they are clearly meaningful entities in context.

Return ONLY the following line-oriented format.

For every entity:

ENTITY: <name> | <type> | <description>

For every explicit relationship:

RELATIONSHIP: <source> | <relation> | <target>

Examples:

ENTITY: Python | TECHNOLOGY | Programming language
ENTITY: Neo4j | TECHNOLOGY | Graph database
ENTITY: Agentic Graph RAG | PROJECT | Retrieval augmented generation architecture
ENTITY: PageRank | METHOD | Graph ranking algorithm

RELATIONSHIP: Agentic Graph RAG | USES | Neo4j
RELATIONSHIP: Agentic Graph RAG | USES | PageRank

Rules:

1. One entity per ENTITY line.
2. One relationship per RELATIONSHIP line.
3. Use the exact entity names from the text.
4. Do not return JSON.
5. Do not use Markdown.
6. Do not add explanations.
7. Do not put multiple entities on one line.
8. Extract entities even if no relationship exists.
9. Prefer specific meaningful entities over generic words.
""".strip()

    logger.info(
        "Starting entity extraction for %d skeletal chunks",
        len(skeletal_chunks),
    )

    for chunk_index, chunk in enumerate(
        skeletal_chunks,
        start=1,
    ):
        # --------------------------------------------------------------
        # Get text safely
        # --------------------------------------------------------------

        text = (
            getattr(
                chunk,
                "enriched_content",
                None,
            )
            or getattr(
                chunk,
                "content",
                None,
            )
            or ""
        )

        text = str(text).strip()

        logger.info(
            "Entity extraction chunk %d/%d: id=%s, text_length=%d",
            chunk_index,
            len(skeletal_chunks),
            chunk.id,
            len(text),
        )

        if not text:
            logger.warning(
                "Skipping entity extraction for chunk %s: empty text",
                chunk.id,
            )
            continue

        # --------------------------------------------------------------
        # Keep enough text for entity extraction
        # --------------------------------------------------------------

        text_for_llm = text[:4000]

        try:
            # ----------------------------------------------------------
            # Use the known-working mini model first.
            # ----------------------------------------------------------

            model = (
                getattr(
                    cfg.openai,
                    "llm_model_mini",
                    None,
                )
                or getattr(
                    cfg.openai,
                    "corrector_model",
                    None,
                )
                or getattr(
                    cfg.openai,
                    "llm_model",
                    None,
                )
            )

            if not model:
                raise RuntimeError(
                    "No LLM model configured for entity extraction"
                )

            logger.info(
                "Entity extraction request: chunk=%s model=%s",
                chunk.id,
                model,
            )

            response = (
                openai_client
                .chat
                .completions
                .create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": text_for_llm,
                        },
                    ],
                    temperature=0.0,
                )
            )

            text_response = (
                response
                .choices[0]
                .message
                .content
                or ""
            )

            text_response = str(
                text_response
            ).strip()

            logger.info(
                "Entity extraction response: chunk=%s response_length=%d",
                chunk.id,
                len(text_response),
            )

            # ----------------------------------------------------------
            # DEBUG: show first part of LLM output
            # ----------------------------------------------------------

            logger.debug(
                "Entity extraction raw response for chunk %s:\n%s",
                chunk.id,
                text_response[:3000],
            )

            # ----------------------------------------------------------
            # Parse response
            # ----------------------------------------------------------

            entities, relationships = (
                _parse_extraction_response(
                    text_response,
                    chunk.id,
                )
            )

            logger.info(
                "Entity extraction result: chunk=%s entities=%d relationships=%d",
                chunk.id,
                len(entities),
                len(relationships),
            )

            all_entities.extend(
                entities
            )

            all_relationships.extend(
                relationships
            )

        except Exception as exc:
            logger.error(
                "Entity extraction failed for chunk %s: %s",
                chunk.id,
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Deduplicate
    # ------------------------------------------------------------------

    entities = deduplicate_entities(
        all_entities
    )

    relationships = deduplicate_relationships(
        all_relationships
    )

    logger.info(
        "Extracted %d unique entities and %d direct relationships "
        "from %d skeletal chunks",
        len(entities),
        len(relationships),
        len(skeletal_chunks),
    )

    return (
        entities,
        relationships,
    )


# ============================================================================
# ROBUST ENTITY EXTRACTION PARSER
# ============================================================================

def _parse_extraction_response(
    text: str,
    source_chunk_id: str,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Parse LLM entity extraction response.

    Primary format:

        ENTITY: name | type | description
        RELATIONSHIP: source | relation | target

    The parser is intentionally tolerant of:
    - Markdown code fences
    - extra whitespace
    - CRLF
    - malformed lines
    """

    entities: list[Entity] = []
    relationships: list[Relationship] = []

    if not text:
        logger.warning(
            "Empty entity extraction response for chunk %s",
            source_chunk_id,
        )
        return [], []

    # --------------------------------------------------------------
    # Remove Markdown fences if model ignored the instruction.
    # --------------------------------------------------------------

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:text|txt|json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    # --------------------------------------------------------------
    # Parse line by line
    # --------------------------------------------------------------

    for raw_line in cleaned.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        # ==========================================================
        # ENTITY
        # ==========================================================

        if line.upper().startswith(
            "ENTITY:"
        ):
            payload = line[
                line.find(":") + 1 :
            ].strip()

            parts = [
                p.strip()
                for p in payload.split("|")
            ]

            # Need at least:
            # name | type
            if len(parts) < 2:
                logger.warning(
                    "Malformed ENTITY line for chunk %s: %s",
                    source_chunk_id,
                    line,
                )
                continue

            name = parts[0]
            entity_type = parts[1]

            description = (
                parts[2]
                if len(parts) >= 3
                else ""
            )

            if not name:
                continue

            # ------------------------------------------------------
            # Deterministic entity ID
            # ------------------------------------------------------

            ent_id = hashlib.md5(
                name.lower()
                .strip()
                .encode("utf-8")
            ).hexdigest()[:8]

            entities.append(
                Entity(
                    id=ent_id,
                    name=name,
                    entity_type=entity_type,
                    description=description,
                    metadata={
                        "source_chunk": source_chunk_id,
                    },
                )
            )

            continue

        # ==========================================================
        # RELATIONSHIP
        # ==========================================================

        if line.upper().startswith(
            "RELATIONSHIP:"
        ):
            payload = line[
                line.find(":") + 1 :
            ].strip()

            parts = [
                p.strip()
                for p in payload.split("|")
            ]

            if len(parts) < 3:
                logger.warning(
                    "Malformed RELATIONSHIP line for chunk %s: %s",
                    source_chunk_id,
                    line,
                )
                continue

            source = parts[0]
            relation = parts[1]
            target = parts[2]

            if not all(
                (
                    source,
                    relation,
                    target,
                )
            ):
                continue

            relation_type = (
                normalize_relation_type(
                    relation
                )
            )

            if not relation_type:
                continue

            relationships.append(
                Relationship(
                    id=_relation_id(
                        source,
                        relation_type,
                        target,
                    ),
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    metadata={
                        "source_chunk": source_chunk_id,
                        "method": "entity_extraction",
                    },
                )
            )

            continue

    # --------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------

    if not entities:
        logger.warning(
            "NO ENTITIES PARSED from chunk %s. "
            "Raw response:\n%s",
            source_chunk_id,
            cleaned[:3000],
        )

    return (
        entities,
        relationships,
    )


# ============================================================================
# 5. CHEAP PERIPHERAL LINKING
# ============================================================================


def link_peripheral_keywords(
    peripheral_chunks: list[Chunk],
    existing_entities: list[Entity],
) -> list[Relationship]:
    """Cheaply connect existing entities to chunks mentioning them."""

    if (
        not peripheral_chunks
        or not existing_entities
    ):
        return []

    relationships: list[
        Relationship
    ] = []

    for chunk in peripheral_chunks:
        text_lower = (
            chunk.enriched_content
            or ""
        ).lower()

        for entity in existing_entities:
            name_lower = (
                entity.name
                .lower()
                .strip()
            )

            if len(name_lower) < 2:
                continue

            if name_lower in text_lower:
                relationships.append(
                    Relationship(
                        id=_relation_id(
                            entity.name,
                            "MENTIONED_IN",
                            chunk.id,
                        ),
                        source=entity.name,
                        target=chunk.id,
                        relation_type=(
                            "MENTIONED_IN"
                        ),
                        metadata={
                            "method": "keyword",
                            "source_chunk": (
                                chunk.id
                            ),
                        },
                    )
                )

    return deduplicate_relationships(
        relationships
    )


# ============================================================================
# 6. ENTITY NORMALIZATION
# ============================================================================


def normalize_entity_name(
    name: str,
) -> str:
    """Normalize an entity name for matching."""

    value = re.sub(
        r"\s+",
        " ",
        (name or "").strip().lower(),
    )

    return value


def normalize_relation_type(
    value: str,
) -> str:
    """Normalize relation type into a safe Neo4j identifier."""

    value = (
        value
        or ""
    ).strip().upper()

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

    return value


def deduplicate_entities(
    entities: list[Entity],
) -> list[Entity]:
    """Merge duplicate entities by normalized name."""

    unique: dict[
        str,
        Entity,
    ] = {}

    for entity in entities:
        key = normalize_entity_name(
            entity.name
        )

        if not key:
            continue

        if key not in unique:
            unique[key] = entity
            continue

        existing = unique[key]

        if (
            not existing.description
            and entity.description
        ):
            existing.description = (
                entity.description
            )

        metadata = dict(
            existing.metadata
            or {}
        )

        other_metadata = dict(
            entity.metadata
            or {}
        )

        for k, v in other_metadata.items():
            if k not in metadata:
                metadata[k] = v

        existing.metadata = metadata

    return list(
        unique.values()
    )


def deduplicate_relationships(
    relationships: list[Relationship],
) -> list[Relationship]:
    """Merge identical typed relationships and aggregate evidence."""

    unique: dict[
        tuple[str, str, str],
        Relationship,
    ] = {}

    for rel in relationships:
        relation_type = (
            normalize_relation_type(
                rel.relation_type
            )
        )

        key = (
            normalize_entity_name(
                rel.source
            ),
            relation_type,
            normalize_entity_name(
                rel.target
            ),
        )

        if not all(key):
            continue

        if key not in unique:
            rel.relation_type = (
                relation_type
            )
            unique[key] = rel
            continue

        existing = unique[key]

        existing_meta = dict(
            existing.metadata
            or {}
        )

        new_meta = dict(
            rel.metadata
            or {}
        )

        old_evidence = int(
            existing_meta.get(
                "evidence_count",
                1,
            )
        )

        new_evidence = int(
            new_meta.get(
                "evidence_count",
                1,
            )
        )

        existing_meta[
            "evidence_count"
        ] = (
            old_evidence
            + new_evidence
        )

        if (
            "confidence" in new_meta
            and "confidence"
            in existing_meta
        ):
            existing_meta[
                "confidence"
            ] = max(
                float(
                    existing_meta[
                        "confidence"
                    ]
                ),
                float(
                    new_meta[
                        "confidence"
                    ]
                ),
            )

        elif "confidence" in new_meta:
            existing_meta[
                "confidence"
            ] = new_meta[
                "confidence"
            ]

        sources: set[str] = set()

        for metadata in (
            existing_meta,
            new_meta,
        ):
            source = metadata.get(
                "source_chunk"
            )

            if source:
                sources.add(
                    str(source)
                )

            for item in metadata.get(
                "source_chunks",
                [],
            ):
                sources.add(
                    str(item)
                )

        if sources:
            existing_meta[
                "source_chunks"
            ] = sorted(
                sources
            )

        existing.metadata = (
            existing_meta
        )

    return list(
        unique.values()
    )


# ============================================================================
# 7. CHEAP ENTITY-PAIR CANDIDATE GENERATION
# ============================================================================


def build_entity_pair_candidates(
    chunks: list[Chunk],
    entities: list[Entity],
    *,
    max_pairs: int | None = None,
    min_cooccurrence: int | None = None,
    context_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Find entity pairs that repeatedly occur in the same chunks.

    IMPORTANT:
        No LLM is used here.

    This is the cheap discovery stage.

    Example:

        10 documents
        500 chunks
        200 entities

    Instead of asking an LLM about all possible pairs:

        200 * 199 / 2 = 19,900 pairs

    we first find only pairs that actually co-occur in the corpus.
    """

    max_pairs = (
        max_pairs
        if max_pairs is not None
        else _cfg_value(
            "semantic_relation_max_pairs",
            500,
        )
    )

    min_cooccurrence = (
        min_cooccurrence
        if min_cooccurrence is not None
        else _cfg_value(
            "semantic_relation_min_cooccurrence",
            1,
        )
    )

    context_chars = (
        context_chars
        if context_chars is not None
        else _cfg_value(
            "semantic_relation_context_chars",
            800,
        )
    )

    entity_by_name: dict[
        str,
        Entity,
    ] = {}

    for entity in deduplicate_entities(
        entities
    ):
        entity_by_name[
            normalize_entity_name(
                entity.name
            )
        ] = entity

    pair_data: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for chunk in chunks:
        text = (
            chunk.enriched_content
            or ""
        )

        text_lower = text.lower()

        found: list[Entity] = []

        for (
            key,
            entity,
        ) in entity_by_name.items():

            if len(key) < 2:
                continue

            if key in text_lower:
                found.append(
                    entity
                )

        if len(found) < 2:
            continue

        found = deduplicate_entities(
            found
        )

        for i in range(
            len(found)
        ):
            for j in range(
                i + 1,
                len(found),
            ):
                a = found[i]
                b = found[j]

                a_key = (
                    normalize_entity_name(
                        a.name
                    )
                )

                b_key = (
                    normalize_entity_name(
                        b.name
                    )
                )

                pair_key = tuple(
                    sorted(
                        (
                            a_key,
                            b_key,
                        )
                    )
                )

                item = (
                    pair_data.setdefault(
                        pair_key,
                        {
                            "source": (
                                a.name
                                if a_key
                                == pair_key[0]
                                else b.name
                            ),
                            "target": (
                                b.name
                                if b_key
                                == pair_key[1]
                                else a.name
                            ),
                            "evidence_count": 0,
                            "contexts": [],
                            "source_chunks": [],
                        },
                    )
                )

                item[
                    "evidence_count"
                ] += 1

                item[
                    "source_chunks"
                ].append(
                    chunk.id
                )

                context = (
                    _build_pair_context(
                        text,
                        a.name,
                        b.name,
                        context_chars,
                    )
                )

                if context:
                    item[
                        "contexts"
                    ].append(
                        context
                    )

    candidates = [
        item
        for item in pair_data.values()
        if item[
            "evidence_count"
        ]
        >= min_cooccurrence
    ]

    candidates.sort(
        key=lambda item: (
            item[
                "evidence_count"
            ],
            len(
                item[
                    "source_chunks"
                ]
            ),
        ),
        reverse=True,
    )

    for candidate in candidates:
        candidate[
            "contexts"
        ] = list(
            dict.fromkeys(
                candidate[
                    "contexts"
                ]
            )
        )[:3]

        candidate[
            "source_chunks"
        ] = list(
            dict.fromkeys(
                candidate[
                    "source_chunks"
                ]
            )
        )

    return candidates[
        :max_pairs
    ]


def _build_pair_context(
    text: str,
    source: str,
    target: str,
    max_chars: int,
) -> str:
    """Extract a small local context around an entity pair."""

    lower = text.lower()

    source_pos = lower.find(
        source.lower()
    )

    target_pos = lower.find(
        target.lower()
    )

    positions = [
        p
        for p in (
            source_pos,
            target_pos,
        )
        if p >= 0
    ]

    if not positions:
        return text[:max_chars]

    center = min(
        positions
    )

    start = max(
        0,
        center - max_chars // 3,
    )

    return text[
        start : start + max_chars
    ].strip()


# ============================================================================
# 8. SEMANTIC RELATION EXTRACTION
# ============================================================================


def extract_semantic_relations(
    candidates: list[dict[str, Any]],
    openai_client: OpenAI | None = None,
    *,
    batch_size: int | None = None,
    allowed_relation_types: list[str] | None = None,
    model: str | None = None,
) -> list[Relationship]:
    """Extract multiple semantic relations between known entity pairs.

    There are two modes.

    MODE A — controlled vocabulary
        semantic_relation_allowlist_enabled = True

        The LLM can return ONLY relations from the approved vocabulary.

    MODE B — discovery
        semantic_relation_allowlist_enabled = False

        The LLM is allowed to propose relation names itself.

        This mode is intended for TEST DISCOVERY / human review.

        After human review, the approved vocabulary can be placed into
        runtime.env and MODE A can be enabled.

    In BOTH modes:
        - source entity must already exist;
        - target entity must already exist;
        - no new entities may be created;
        - multiple relations per pair are allowed.
    """

    cfg = get_settings()

    if not candidates:
        return []

    if not _cfg_value(
        "semantic_relations_enabled",
        True,
    ):
        return []

    allowlist_enabled = bool(
        _cfg_value(
            "semantic_relation_allowlist_enabled",
            False,
        )
    )

    if allowed_relation_types is None:
        try:
            allowed_relation_types = (
                cfg.indexing.semantic_relation_type_list()
            )
        except AttributeError:
            allowed_relation_types = []

    allowed = {
        normalize_relation_type(
            value
        )
        for value in (
            allowed_relation_types
            or []
        )
        if normalize_relation_type(
            value
        )
    }

    # If the allow-list is explicitly enabled, it MUST NOT be empty.
    if (
        allowlist_enabled
        and not allowed
    ):
        logger.warning(
            "Semantic relation extraction skipped: "
            "allow-list mode is enabled but no relation types exist."
        )
        return []

    if batch_size is None:
        batch_size = _cfg_value(
            "semantic_relation_batch_size",
            10,
        )

    batch_size = max(
        1,
        int(batch_size),
    )

    if model is None or not model.strip():
        model = (
            _cfg_value(
                "semantic_relation_model",
                "",
            )
            or cfg.openai.corrector_model
        )

    if openai_client is None:
        from rag_core.config import (
            make_openai_client,
        )

        openai_client = (
            make_openai_client(
                cfg
            )
        )

    all_relationships: list[
        Relationship
    ] = []

    for start in range(
        0,
        len(candidates),
        batch_size,
    ):
        batch = candidates[
            start : start + batch_size
        ]

        try:
            extracted = (
                _extract_relation_batch(
                    batch,
                    openai_client,
                    model=model,
                    allowed_relation_types=(
                        sorted(allowed)
                    ),
                    allowlist_enabled=(
                        allowlist_enabled
                    ),
                    min_confidence=float(
                        _cfg_value(
                            "semantic_relation_min_confidence",
                            0.60,
                        )
                    ),
                    max_per_pair=int(
                        _cfg_value(
                            "semantic_relation_max_per_pair",
                            5,
                        )
                    ),
                )
            )

            all_relationships.extend(
                extracted
            )

        except Exception as exc:
            logger.error(
                "Semantic relation batch failed "
                "(batch %d-%d): %s",
                start,
                start + len(batch),
                exc,
                exc_info=True,
            )

    result = deduplicate_relationships(
        all_relationships
    )

    logger.info(
        "Semantic relation extraction: "
        "%d candidate pairs -> "
        "%d unique semantic relationships",
        len(candidates),
        len(result),
    )

    return result


def _extract_relation_batch(
    candidates: list[dict[str, Any]],
    client: OpenAI,
    *,
    model: str,
    allowed_relation_types: list[str],
    allowlist_enabled: bool,
    min_confidence: float,
    max_per_pair: int,
) -> list[Relationship]:
    """Run one batched LLM relation extraction request."""

    payload_candidates = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        payload_candidates.append(
            {
                "id": index,
                "source": candidate[
                    "source"
                ],
                "target": candidate[
                    "target"
                ],
                "evidence_count": candidate[
                    "evidence_count"
                ],
                "contexts": candidate[
                    "contexts"
                ],
            }
        )

    if allowlist_enabled:
        relation_instruction = (
            "You may use ONLY these approved "
            "relation types:\n"
            + ", ".join(
                allowed_relation_types
            )
        )
    else:
        relation_instruction = (
            "There is NO predefined relation "
            "dictionary yet.\n"
            "You may propose a concise, reusable "
            "semantic relation type that best describes "
            "the evidence.\n"
            "Use a short UPPER_SNAKE_CASE identifier.\n"
            "Examples of the FORMAT only: "
            "LOCATED_IN, PART_OF, CAUSES, "
            "CONNECTED_TO, USES, PRODUCES, "
            "DEPENDS_ON, PARTICIPATES_IN.\n"
            "Do not restrict yourself to these examples."
        )

    system_prompt = f"""
You are a conservative knowledge-graph relation discovery engine.

IMPORTANT:
You are NOT allowed to discover new entities.

Every source and target MUST be copied from the candidate pair exactly.

Your task is to determine whether one or more meaningful semantic
relationships exist between the two already-known entities.

{relation_instruction}

A single pair MAY have multiple relations when the evidence supports them.

Example:

Liver -> LOCATED_IN -> Abdominal_Cavity
Liver -> CONNECTED_TO -> Portal_Vein
Liver -> PARTICIPATES_IN -> Blood_Circulation
Liver -> PARTICIPATES_IN -> Metabolism

Do NOT create a relation merely because two entities occur together.

Co-occurrence is evidence for being a candidate,
NOT proof of a semantic relationship.

Do NOT invent causal relationships.

Do NOT use vague relations such as:
RELATED_TO
MENTIONED_WITH
SAME_TOPIC

unless the evidence genuinely supports such a relation.

Prefer specific reusable semantic relations.

Return JSON only:

{{
  "relations": [
    {{
      "candidate_id": 1,
      "source": "exact candidate source",
      "relation": "RELATION_TYPE",
      "target": "exact candidate target",
      "confidence": 0.0,
      "evidence": "short evidence statement"
    }}
  ]
}}

Confidence must be between 0 and 1.

Only return relations with confidence >=
{min_confidence:.2f}.

Return at most {max_per_pair} relations for each candidate pair.
""".strip()

    user_prompt = json.dumps(
        {
            "candidates": payload_candidates,
        },
        ensure_ascii=False,
        indent=2,
    )

    response = (
        client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=float(
                _cfg_value(
                    "semantic_relation_temperature",
                    0.0,
                )
            ),
        )
    )

    text = (
        response.choices[0]
        .message.content
        or ""
    )

    data = _parse_json_object(
        text
    )

    if not isinstance(
        data,
        dict,
    ):
        return []

    raw_relations = data.get(
        "relations",
        [],
    )

    if not isinstance(
        raw_relations,
        list,
    ):
        return []

    by_id = {
        index: candidate
        for index, candidate in enumerate(
            candidates,
            start=1,
        )
    }

    relationships: list[
        Relationship
    ] = []

    for item in raw_relations:
        if not isinstance(
            item,
            dict,
        ):
            continue

        try:
            candidate_id = int(
                item.get(
                    "candidate_id",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        candidate = by_id.get(
            candidate_id
        )

        if candidate is None:
            continue

        source = str(
            item.get(
                "source",
                "",
            )
        ).strip()

        target = str(
            item.get(
                "target",
                "",
            )
        ).strip()

        relation = (
            normalize_relation_type(
                str(
                    item.get(
                        "relation",
                        "",
                    )
                )
            )
        )

        try:
            confidence = float(
                item.get(
                    "confidence",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        # Source must be the known source entity.
        if (
            normalize_entity_name(
                source
            )
            != normalize_entity_name(
                candidate["source"]
            )
        ):
            continue

        # Target must be the known target entity.
        if (
            normalize_entity_name(
                target
            )
            != normalize_entity_name(
                candidate["target"]
            )
        ):
            continue

        # Relation must be valid.
        if not relation:
            continue

        # In controlled mode, relation must be approved.
        if (
            allowlist_enabled
            and relation
            not in allowed_relation_types
        ):
            continue

        if confidence < min_confidence:
            continue

        evidence = str(
            item.get(
                "evidence",
                "",
            )
        ).strip()

        relationships.append(
            Relationship(
                id=_relation_id(
                    source,
                    relation,
                    target,
                ),
                source=source,
                target=target,
                relation_type=relation,
                metadata={
                    "method": (
                        "semantic_relation_batch"
                    ),
                    "confidence": confidence,
                    "evidence": evidence,
                    "evidence_count": candidate[
                        "evidence_count"
                    ],
                    "source_chunks": candidate[
                        "source_chunks"
                    ],
                },
            )
        )

    return relationships


def _parse_json_object(
    text: str,
) -> dict[str, Any] | None:
    """Parse JSON even if the LLM wrapped it in markdown fences."""

    text = (
        text
        or ""
    ).strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    try:
        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if start < 0 or end <= start:
        return None

    try:
        parsed = json.loads(
            text[
                start : end + 1
            ]
        )

        return (
            parsed
            if isinstance(
                parsed,
                dict,
            )
            else None
        )

    except json.JSONDecodeError:
        return None


# ============================================================================
# 9. CORPUS DISCOVERY — NO LLM
# ============================================================================


def collect_graph_relation_candidates(
    driver: Driver,
    *,
    max_pairs: int | None = None,
    sample_contexts: int | None = None,
    min_evidence: int | None = None,
) -> list[dict[str, Any]]:
    """Find frequently co-mentioned PhraseNode pairs.

    THIS FUNCTION DOES NOT USE AN LLM.

    It is the cheapest way to inspect a large corpus.

    Example:

        Entity A + Entity B
        appeared together in 17 passages.

    This pair becomes more interesting than:

        Entity C + Entity D
        appeared together once.
    """

    max_pairs = (
        max_pairs
        if max_pairs is not None
        else _cfg_value(
            "discovery_max_pairs",
            500,
        )
    )

    sample_contexts = (
        sample_contexts
        if sample_contexts is not None
        else _cfg_value(
            "discovery_sample_contexts",
            3,
        )
    )

    min_evidence = (
        min_evidence
        if min_evidence is not None
        else _cfg_value(
            "discovery_min_evidence",
            2,
        )
    )

    query = """
    MATCH (a:PhraseNode)-[]->(p:PassageNode)<-[]-(b:PhraseNode)
    WHERE a.name IS NOT NULL
      AND b.name IS NOT NULL
      AND a.name <> b.name
      AND toLower(a.name) < toLower(b.name)

    WITH
        a.name AS source,
        b.name AS target,
        collect(DISTINCT p.content)[0..5] AS contexts,
        count(DISTINCT p) AS evidence_count

    WHERE evidence_count >= $min_evidence

    RETURN
        source,
        target,
        evidence_count,
        contexts

    ORDER BY evidence_count DESC

    LIMIT $max_pairs
    """

    candidates: list[
        dict[str, Any]
    ] = []

    with driver.session() as session:
        result = session.run(
            query,
            min_evidence=int(
                min_evidence
            ),
            max_pairs=int(
                max_pairs
            ),
        )

        for record in result:
            contexts = [
                str(value)
                for value in (
                    record[
                        "contexts"
                    ]
                    or []
                )
                if value
            ]

            candidates.append(
                {
                    "source": record[
                        "source"
                    ],
                    "target": record[
                        "target"
                    ],
                    "evidence_count": int(
                        record[
                            "evidence_count"
                        ]
                    ),
                    "contexts": contexts[
                        :sample_contexts
                    ],
                    "source_chunks": [],
                }
            )

    logger.info(
        "Discovery found %d candidate entity pairs",
        len(candidates),
    )

    return candidates


# ============================================================================
# 10. RELATION FREQUENCY REPORT
# ============================================================================


def relation_frequency_report(
    driver: Driver,
) -> list[dict[str, Any]]:
    """Count semantic relationships already stored in Neo4j."""

    query = """
    MATCH (a:PhraseNode)-[r]->(b:PhraseNode)
    RETURN
        type(r) AS relation,
        count(r) AS count
    ORDER BY count DESC
    """

    rows: list[
        dict[str, Any]
    ] = []

    with driver.session() as session:
        for record in session.run(
            query
        ):
            rows.append(
                {
                    "relation": record[
                        "relation"
                    ],
                    "count": int(
                        record[
                            "count"
                        ]
                    ),
                }
            )

    return rows


# ============================================================================
# 11. DISCOVERY REPORT — FREQUENCY WITHOUT LLM
# ============================================================================


def build_discovery_report(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert raw candidate pairs into a human-readable report.

    This is useful for the Streamlit Discovery tab.

    No LLM is called.
    """

    report = []

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        report.append(
            {
                "rank": rank,
                "source": candidate[
                    "source"
                ],
                "target": candidate[
                    "target"
                ],
                "evidence_count": candidate[
                    "evidence_count"
                ],
                "contexts": candidate[
                    "contexts"
                ],
            }
        )

    return report


# ============================================================================
# 12. APPLY RELATIONSHIPS TO NEO4J
# ============================================================================


def apply_relationships_to_neo4j(
    driver: Driver,
    relationships: list[Relationship],
) -> int:
    """Create typed PhraseNode -> PhraseNode edges in Neo4j."""

    relationships = (
        deduplicate_relationships(
            relationships
        )
    )

    applied = 0

    for rel in relationships:
        relation_type = (
            normalize_relation_type(
                rel.relation_type
            )
        )

        if not relation_type:
            continue

        if not re.fullmatch(
            r"[A-Z][A-Z0-9_]*",
            relation_type,
        ):
            continue

        metadata = dict(
            rel.metadata
            or {}
        )

        query = f"""
        MATCH (a:PhraseNode)
        WHERE toLower(a.name) = toLower($source)

        MATCH (b:PhraseNode)
        WHERE toLower(b.name) = toLower($target)

        MERGE (a)-[r:{relation_type}]->(b)

        SET
            r.confidence =
                CASE
                    WHEN $confidence IS NULL
                        THEN coalesce(
                            r.confidence,
                            0.0
                        )

                    WHEN coalesce(
                        r.confidence,
                        0.0
                    ) > $confidence
                        THEN coalesce(
                            r.confidence,
                            0.0
                        )

                    ELSE $confidence
                END,

            r.evidence_count =
                coalesce(
                    r.evidence_count,
                    0
                )
                + $evidence_count,

            r.discovery_method =
                $method,

            r.evidence =
                CASE
                    WHEN $evidence = ""
                        THEN coalesce(
                            r.evidence,
                            ""
                        )
                    ELSE $evidence
                END,

            r.source_chunks =
                $source_chunks

        RETURN count(r) AS count
        """

        with driver.session() as session:
            result = session.run(
                query,
                source=rel.source,
                target=rel.target,
                confidence=(
                    float(
                        metadata[
                            "confidence"
                        ]
                    )
                    if metadata.get(
                        "confidence"
                    )
                    is not None
                    else None
                ),
                evidence_count=int(
                    metadata.get(
                        "evidence_count",
                        1,
                    )
                ),
                method=str(
                    metadata.get(
                        "method",
                        "semantic_discovery",
                    )
                ),
                evidence=str(
                    metadata.get(
                        "evidence",
                        "",
                    )
                ),
                source_chunks=[
                    str(x)
                    for x in metadata.get(
                        "source_chunks",
                        [],
                    )
                ],
            )

            record = result.single()

            if (
                record
                and int(
                    record["count"]
                )
                > 0
            ):
                applied += 1

    logger.info(
        "Applied %d semantic relationships to Neo4j",
        applied,
    )

    return applied


# ============================================================================
# 13. COMPLETE CORPUS DISCOVERY
# ============================================================================


def discover_and_extract_relations(
    driver: Driver,
    openai_client: OpenAI | None = None,
) -> dict[str, Any]:
    """Run the complete corpus discovery pipeline.

    Stage 1 — FREE
        Find repeated entity pairs.

    Stage 2 — OPTIONAL LLM
        Analyse only the selected candidate pairs.

    Stage 3 — HUMAN REVIEW
        The returned relations can be displayed in Streamlit.

    Stage 4 — OPTIONAL WRITE
        Approved relations can then be written to Neo4j.

    The function intentionally DOES NOT automatically write anything
    to Neo4j.
    """

    cfg = get_settings()

    candidates = (
        collect_graph_relation_candidates(
            driver,
            max_pairs=_cfg_value(
                "discovery_max_pairs",
                500,
            ),
            sample_contexts=_cfg_value(
                "discovery_sample_contexts",
                3,
            ),
            min_evidence=_cfg_value(
                "discovery_min_evidence",
                2,
            ),
        )
    )

    batch_size = max(
        1,
        int(
            _cfg_value(
                "discovery_batch_size",
                10,
            )
        ),
    )

    result: dict[
        str,
        Any,
    ] = {
        "candidates": candidates,
        "relations": [],
        "candidate_count": len(
            candidates
        ),
        "llm_calls_estimate": 0,
        "llm_enabled": bool(
            _cfg_value(
                "discovery_llm_enabled",
                False,
            )
        ),
    }

    if not candidates:
        return result

    if not result[
        "llm_enabled"
    ]:
        return result

    result[
        "llm_calls_estimate"
    ] = (
        len(candidates)
        + batch_size
        - 1
    ) // batch_size

    relations = (
        extract_semantic_relations(
            candidates,
            openai_client=openai_client,
            batch_size=batch_size,
        )
    )

    result[
        "relations"
    ] = relations

    return result


# ============================================================================
# 14. MAIN SKELETON PIPELINE
# ============================================================================


def build_skeleton_index(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    openai_client: OpenAI | None = None,
) -> tuple[
    list[Entity],
    list[Relationship],
    list[Chunk],
    list[Chunk],
]:
    """Build KET-RAG skeleton index."""

    if not chunks:
        logger.warning(
            "Skeleton index: 0 chunks"
        )
        return [], [], [], []

    if not embeddings:
        logger.warning(
            "Skeleton index: 0 embeddings"
        )
        return [], [], [], []

    if len(chunks) != len(embeddings):
        logger.error(
            "Skeleton index: chunks/embeddings mismatch: "
            "chunks=%d embeddings=%d",
            len(chunks),
            len(embeddings),
        )
        return [], [], [], []

    # ==================================================================
    # 1. KNN
    # ==================================================================

    knn_graph = build_knn_graph(
        chunks,
        embeddings,
    )

    # ==================================================================
    # 2. PageRank
    # ==================================================================

    pagerank_scores = compute_pagerank(
        knn_graph
    )

    # ==================================================================
    # 3. Skeleton selection
    # ==================================================================

    skeletal, peripheral = (
        select_skeletal_chunks(
            chunks,
            pagerank_scores,
        )
    )

    logger.info(
        "SKELETON DEBUG: total_chunks=%d skeletal=%d peripheral=%d",
        len(chunks),
        len(skeletal),
        len(peripheral),
    )

    # ==================================================================
    # 4. Entity extraction
    # ==================================================================

    entities, relationships = (
        extract_entities_full(
            skeletal,
            openai_client,
        )
    )

    logger.info(
        "SKELETON DEBUG AFTER ENTITY EXTRACTION: "
        "entities=%d relationships=%d",
        len(entities),
        len(relationships),
    )

    # ==================================================================
    # 5. Peripheral mentions
    # ==================================================================

    peripheral_rels = (
        link_peripheral_keywords(
            peripheral,
            entities,
        )
    )

    relationships.extend(
        peripheral_rels
    )

    # ==================================================================
    # 6. Semantic candidates
    # ==================================================================

    candidates = (
        build_entity_pair_candidates(
            chunks,
            entities,
            max_pairs=_cfg_value(
                "semantic_relation_max_pairs",
                500,
            ),
            min_cooccurrence=_cfg_value(
                "semantic_relation_min_cooccurrence",
                1,
            ),
            context_chars=_cfg_value(
                "semantic_relation_context_chars",
                800,
            ),
        )
    )

    logger.info(
        "SKELETON DEBUG: semantic candidates=%d",
        len(candidates),
    )

    # ==================================================================
    # 7. Semantic relations
    # ==================================================================

    semantic_relations: list[
        Relationship
    ] = []

    if _cfg_value(
        "semantic_relations_enabled",
        True,
    ):
        semantic_relations = (
            extract_semantic_relations(
                candidates,
                openai_client=openai_client,
                batch_size=_cfg_value(
                    "semantic_relation_batch_size",
                    10,
                ),
            )
        )

    relationships.extend(
        semantic_relations
    )

    # ==================================================================
    # 8. Deduplicate
    # ==================================================================

    entities = deduplicate_entities(
        entities
    )

    relationships = deduplicate_relationships(
        relationships
    )

    logger.info(
        "Skeleton index built: "
        "%d entities, "
        "%d relationships "
        "(%d skeletal + %d peripheral chunks, "
        "%d semantic candidates)",
        len(entities),
        len(relationships),
        len(skeletal),
        len(peripheral),
        len(candidates),
    )

    return (
        entities,
        relationships,
        skeletal,
        peripheral,
    )


# ============================================================================
# 15. RELATION ID
# ============================================================================


def _relation_id(
    source: str,
    relation_type: str,
    target: str,
) -> str:
    """Create deterministic relation ID."""

    return hashlib.md5(
        (
            f"{normalize_entity_name(source)}:"
            f"{normalize_relation_type(relation_type)}:"
            f"{normalize_entity_name(target)}"
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:8]
