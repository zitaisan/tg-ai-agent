"""
Semantic Relationship Discovery.

Discovers candidate semantic relationships between entities.

Methods:

1. HEURISTIC
   - no LLM calls
   - cheap rule-based discovery
   - supports Russian and English relation patterns
   - extracts only normalized semantic predicates

2. LLM
   - semantic relationship extraction
   - processes several chunks per LLM call
   - produces short normalized predicates

3. BOTH
   - runs heuristic + LLM
   - aggregates candidates
   - useful for corpus-level discovery

Important:
    Heuristic discovery is intentionally conservative.
    It does NOT treat arbitrary words between two entities as a relation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_core.config import get_settings
from rag_core.models import Chunk, Entity, Relationship

from agentic_graph_rag.indexing.dual_node import create_phrase_relationships


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

MAX_RELATION_WORDS = 5
MAX_RELATION_LENGTH = 50
DEFAULT_HEURISTIC_WINDOW = 250


# ============================================================================
# Canonical relation aliases
# ============================================================================

_RELATION_ALIASES: dict[str, str] = {

    # USES
    "использует": "USES",
    "используют": "USES",
    "использовал": "USES",
    "использовала": "USES",
    "использовали": "USES",
    "используется": "USES",
    "используются": "USES",
    "применяет": "USES",
    "применяют": "USES",
    "применяется": "USES",
    "применяются": "USES",

    "uses": "USES",
    "use": "USES",
    "using": "USES",
    "utilizes": "USES",
    "utilize": "USES",
    "applies": "USES",
    "applied": "USES",

    # IMPLEMENTS
    "реализует": "IMPLEMENTS",
    "реализуют": "IMPLEMENTS",
    "реализуется": "IMPLEMENTS",
    "реализуются": "IMPLEMENTS",

    "implements": "IMPLEMENTS",
    "implement": "IMPLEMENTS",
    "implemented": "IMPLEMENTS",

    # PROVIDES
    "предоставляет": "PROVIDES",
    "предоставляют": "PROVIDES",
    "предоставляет возможность": "PROVIDES",
    "обеспечивает": "PROVIDES",
    "обеспечивают": "PROVIDES",

    "provides": "PROVIDES",
    "provide": "PROVIDES",
    "provided": "PROVIDES",

    # ENABLES
    "позволяет": "ENABLES",
    "позволяют": "ENABLES",
    "позволяет использовать": "ENABLES",
    "дает возможность": "ENABLES",
    "даёт возможность": "ENABLES",

    "enables": "ENABLES",
    "enable": "ENABLES",
    "allows": "ENABLES",
    "allow": "ENABLES",

    # CONTAINS
    "содержит": "CONTAINS",
    "содержат": "CONTAINS",
    "содержащий": "CONTAINS",
    "включает": "CONTAINS",
    "включают": "CONTAINS",
    "включает в себя": "CONTAINS",
    "включают в себя": "CONTAINS",

    "contains": "CONTAINS",
    "contain": "CONTAINS",
    "includes": "CONTAINS",
    "include": "CONTAINS",

    # PART_OF
    "является частью": "PART_OF",
    "являются частью": "PART_OF",
    "является частью системы": "PART_OF",
    "является компонентом": "PART_OF",
    "являются компонентами": "PART_OF",
    "входит в состав": "PART_OF",
    "входит в": "PART_OF",
    "входят в": "PART_OF",

    "is part of": "PART_OF",
    "are part of": "PART_OF",
    "belongs to": "PART_OF",
    "is a component of": "PART_OF",

    # COMPOSED_OF
    "состоит из": "COMPOSED_OF",
    "состоящий из": "COMPOSED_OF",
    "состоящая из": "COMPOSED_OF",
    "состоят из": "COMPOSED_OF",

    "consists of": "COMPOSED_OF",
    "composed of": "COMPOSED_OF",
    "made of": "COMPOSED_OF",

    # DEPENDS_ON
    "зависит от": "DEPENDS_ON",
    "зависят от": "DEPENDS_ON",
    "зависимость от": "DEPENDS_ON",

    "depends on": "DEPENDS_ON",
    "dependent on": "DEPENDS_ON",

    # BASED_ON
    "основан на": "BASED_ON",
    "основана на": "BASED_ON",
    "основано на": "BASED_ON",
    "основывается на": "BASED_ON",
    "базируется на": "BASED_ON",

    "based on": "BASED_ON",

    # BUILDS_ON
    "построен на": "BUILDS_ON",
    "построена на": "BUILDS_ON",
    "построено на": "BUILDS_ON",

    "builds on": "BUILDS_ON",

    # BUILDS
    "строит": "BUILDS",
    "строится": "BUILDS",
    "создает": "BUILDS",
    "создаёт": "BUILDS",
    "создают": "BUILDS",
    "формирует": "BUILDS",
    "формируют": "BUILDS",

    "builds": "BUILDS",
    "build": "BUILDS",
    "creates": "BUILDS",
    "constructs": "BUILDS",

    # EXTENDS
    "расширяет": "EXTENDS",
    "расширяет возможности": "EXTENDS",

    "extends": "EXTENDS",
    "extension of": "EXTENDS",

    # IMPROVES
    "улучшает": "IMPROVES",
    "улучшает качество": "IMPROVES",
    "повышает": "IMPROVES",
    "повышает качество": "IMPROVES",

    "improves": "IMPROVES",
    "improve": "IMPROVES",
    "enhances": "IMPROVES",

    # SOLVES
    "решает": "SOLVES",
    "решает проблему": "SOLVES",
    "устраняет": "SOLVES",

    "solves": "SOLVES",
    "addresses": "ADDRESSES",

    # DEFINES
    "определяет": "DEFINES",
    "определяют": "DEFINES",
    "задает": "DEFINES",
    "задаёт": "DEFINES",

    "defines": "DEFINES",
    "define": "DEFINES",

    # EXTRACTS
    "извлекает": "EXTRACTS",
    "извлекают": "EXTRACTS",
    "извлекается": "EXTRACTS",

    "extracts": "EXTRACTS",
    "extract": "EXTRACTS",

    # CLASSIFIES
    "классифицирует": "CLASSIFIES",
    "классифицируют": "CLASSIFIES",

    "classifies": "CLASSIFIES",
    "classify": "CLASSIFIES",

    # SELECTS
    "выбирает": "SELECTS",
    "выбирают": "SELECTS",
    "отбирает": "SELECTS",
    "отбирают": "SELECTS",

    "selects": "SELECTS",
    "select": "SELECTS",

    # CONNECTS
    "соединяет": "CONNECTS",
    "соединяет с": "CONNECTS",
    "связывает": "CONNECTS",
    "связывает с": "CONNECTS",

    "connects": "CONNECTS",

    # CONNECTED_TO
    "соединены": "CONNECTED_TO",
    "связаны": "CONNECTED_TO",
    "connected to": "CONNECTED_TO",

    # ASSOCIATED_WITH
    "связан с": "ASSOCIATED_WITH",
    "связана с": "ASSOCIATED_WITH",
    "связаны с": "ASSOCIATED_WITH",
    "связан": "ASSOCIATED_WITH",
    "associated with": "ASSOCIATED_WITH",

    # LOCATED_IN
    "находится в": "LOCATED_IN",
    "находится внутри": "LOCATED_IN",
    "расположен в": "LOCATED_IN",
    "расположена в": "LOCATED_IN",
    "расположено в": "LOCATED_IN",
    "расположены в": "LOCATED_IN",

    "located in": "LOCATED_IN",
    "situated in": "LOCATED_IN",

    # PRODUCES
    "производит": "PRODUCES",
    "производят": "PRODUCES",
    "генерирует": "PRODUCES",
    "генерируют": "PRODUCES",

    "produces": "PRODUCES",
    "generates": "PRODUCES",

    # REQUIRES
    "требует": "REQUIRES",
    "требуют": "REQUIRES",
    "требуется": "REQUIRES",

    "requires": "REQUIRES",
    "require": "REQUIRES",

    # SUPPORTS
    "поддерживает": "SUPPORTS",
    "поддерживают": "SUPPORTS",
    "поддерживается": "SUPPORTS",

    "supports": "SUPPORTS",
    "support": "SUPPORTS",

    # CAUSES
    "вызывает": "CAUSES",
    "вызывают": "CAUSES",
    "приводит к": "CAUSES",
    "приводят к": "CAUSES",
    "является причиной": "CAUSES",

    "causes": "CAUSES",
    "leads to": "CAUSES",
    "results in": "CAUSES",

    # TREATS
    "лечит": "TREATS",
    "лечат": "TREATS",
    "лечит заболевание": "TREATS",
    "используется для лечения": "TREATS",

    "treats": "TREATS",
    "treat": "TREATS",

    # USED_FOR
    "используется для": "USED_FOR",
    "используется в": "USED_FOR",
    "предназначен для": "USED_FOR",
    "предназначена для": "USED_FOR",
    "применяется для": "USED_FOR",

    "used for": "USED_FOR",
    "used to": "USED_FOR",
    "designed for": "USED_FOR",

    # DERIVED_FROM
    "получен из": "DERIVED_FROM",
    "получена из": "DERIVED_FROM",
    "получено из": "DERIVED_FROM",
    "выведен из": "DERIVED_FROM",
    "происходит из": "DERIVED_FROM",

    "derived from": "DERIVED_FROM",
    "comes from": "DERIVED_FROM",

    # INTRODUCES
    "вводит": "INTRODUCES",
    "представляет": "INTRODUCES",

    "introduces": "INTRODUCES",

    # PROPOSES
    "предлагает": "PROPOSES",
    "proposes": "PROPOSES",

    # COMPARES
    "сравнивает": "COMPARES",
    "сравнивают": "COMPARES",

    "compares": "COMPARES",
}


# ============================================================================
# Invalid relation values
# ============================================================================

_INVALID_EXACT_RELATIONS = {
    "",
    "RELATED",
    "RELATED_TO",
    "RELATION",
    "RELATIONSHIP",
    "THING",
    "THINGS",
    "CONNECTION",
    "CONNECTIONS",
    "STUFF",
    "ASSOCIATED",
    "AND",
    "OR",
    "IS",
    "ARE",
    "WAS",
    "WERE",
    "BE",
    "THIS",
    "THAT",
    "ВО",
    "В",
    "НА",
    "И",
    "ИЛИ",
    "НЕ",
    "ЧТО",
    "ЭТО",
    "КАК",
    "КОГДА",
    "ГДЕ",
    "ПРИ",
}


_INVALID_TOKENS = {
    "КОТОРЫЙ",
    "КОТОРАЯ",
    "КОТОРЫЕ",
    "КОТОРОЕ",
    "КОГДА",
    "ГДЕ",
    "ЕСЛИ",
    "ПОТОМУ",
    "ПОЭТОМУ",
    "КАК",
    "ЧТО",
    "ЭТО",
    "МОЖЕТ",
    "МОЖНО",
    "ЯВЛЯЕТСЯ",
    "ЯВЛЯЮТСЯ",
    "БЫЛ",
    "БЫЛА",
    "БЫЛИ",
    "БУДУТ",
}


# ============================================================================
# Relation normalization
# ============================================================================

def _normalize_relation(text: str) -> str:
    """
    Normalize arbitrary relation text into UPPER_SNAKE_CASE.
    """

    text = str(text or "").strip().lower()

    if not text:
        return ""

    text = re.sub(
        r"[^a-zA-Zа-яА-ЯёЁ0-9\s_-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return ""

    if text in _RELATION_ALIASES:
        return _RELATION_ALIASES[text]

    prefixes = (
        "is ",
        "are ",
        "was ",
        "were ",
        "be ",
        "being ",
        "has ",
        "have ",
        "had ",
        "can ",
        "may ",
        "является ",
        "являются ",
        "это ",
    )

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    if text in _RELATION_ALIASES:
        return _RELATION_ALIASES[text]

    relation = re.sub(
        r"[^a-zA-ZА-ЯЁа-яё0-9]+",
        "_",
        text.upper(),
    ).strip("_")

    return relation


# ============================================================================
# Relation validation
# ============================================================================

def _is_valid_relation_candidate(relation: str) -> bool:
    """
    Strict relation validator.
    """

    relation = str(
        relation or ""
    ).strip().upper()

    if not relation:
        return False

    if relation in _INVALID_EXACT_RELATIONS:
        return False

    tokens = [
        token
        for token in relation.split("_")
        if token
    ]

    if not tokens:
        return False

    if len(tokens) > MAX_RELATION_WORDS:
        return False

    if len(relation) > MAX_RELATION_LENGTH:
        return False

    if any(
        len(token) > 35
        for token in tokens
    ):
        return False

    if any(
        token in _INVALID_TOKENS
        for token in tokens
    ):
        return False

    if any(
        char in relation
        for char in ".!?;,"
    ):
        return False

    return True


# ============================================================================
# Relationship ID
# ============================================================================

def _relation_id(
    source: str,
    relation: str,
    target: str,
) -> str:
    """
    Generate stable relationship ID.
    """

    raw = (
        f"{source.strip().lower()}|"
        f"{relation.strip().lower()}|"
        f"{target.strip().lower()}"
    )

    return hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()[:12]


# ============================================================================
# Approved schema
# ============================================================================

def load_approved_schema() -> dict[str, Any]:
    """
    Load human-approved relationship schema.
    """

    cfg = get_settings()

    path = Path(
        cfg.relationship_discovery.schema_path
    )

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        return {
            "relationships": []
        }

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, dict):
            return {
                "relationships": []
            }

        relationships = data.get(
            "relationships",
            [],
        )

        if not isinstance(
            relationships,
            list,
        ):
            relationships = []

        return {
            "relationships": relationships
        }

    except Exception as exc:
        logger.warning(
            "Could not load relationship schema: %s",
            exc,
        )

        return {
            "relationships": []
        }


def approved_relation_names() -> list[str]:
    """
    Return normalized approved relationship names.
    """

    schema = load_approved_schema()

    result: list[str] = []

    for item in schema.get(
        "relationships",
        [],
    ):

        if not isinstance(item, dict):
            continue

        name = item.get("name")

        if not name:
            continue

        normalized = _normalize_relation(
            str(name)
        )

        if not normalized:
            continue

        if not _is_valid_relation_candidate(
            normalized
        ):
            continue

        result.append(normalized)

    return sorted(set(result))


# ============================================================================
# Entity matching
# ============================================================================

def _find_entity_mentions(
    text: str,
    entities: list[Entity],
) -> list[tuple[int, int, Entity]]:
    """
    Find explicit entity mentions in text.

    Longer entity names are matched first.
    Overlapping mentions are removed.
    """

    mentions: list[
        tuple[int, int, Entity]
    ] = []

    if not text or not entities:
        return mentions

    text_lower = text.lower()

    unique_entities: dict[
        str,
        Entity,
    ] = {}

    for entity in entities:

        name = str(
            entity.name or ""
        ).strip()

        if len(name) < 2:
            continue

        unique_entities[
            name.lower()
        ] = entity

    sorted_entities = sorted(
        unique_entities.values(),
        key=lambda e: len(e.name),
        reverse=True,
    )

    for entity in sorted_entities:

        name = entity.name.strip()

        pattern = re.compile(
            r"(?<!\w)"
            + re.escape(name.lower())
            + r"(?!\w)"
        )

        for match in pattern.finditer(
            text_lower
        ):

            mentions.append(
                (
                    match.start(),
                    match.end(),
                    entity,
                )
            )

    mentions.sort(
        key=lambda x: (
            x[0],
            -(x[1] - x[0]),
        )
    )

    result: list[
        tuple[int, int, Entity]
    ] = []

    occupied_until = -1

    for mention in mentions:

        start = mention[0]
        end = mention[1]

        if start < occupied_until:
            continue

        result.append(mention)
        occupied_until = end

    return result


def _entities_in_chunk(
    chunk: Chunk,
    entities: list[Entity],
) -> list[str]:
    """
    Return canonical entity names explicitly belonging to this chunk.

    Priority:

    1. source_chunk metadata
    2. explicit text matching
    """

    if chunk is None:
        return []

    chunk_id = str(
        getattr(chunk, "id", "")
    )

    result: list[str] = []

    # ------------------------------------------------------------------
    # 1. Metadata-based lookup
    # ------------------------------------------------------------------

    for entity in entities:

        metadata = getattr(
            entity,
            "metadata",
            None,
        ) or {}

        source_chunk = metadata.get(
            "source_chunk"
        )

        if source_chunk is None:
            continue

        if str(source_chunk) != chunk_id:
            continue

        name = str(
            getattr(entity, "name", "")
            or ""
        ).strip()

        if name:
            result.append(name)

    if result:
        return sorted(
            set(result),
            key=str.lower,
        )

    # ------------------------------------------------------------------
    # 2. Text fallback
    # ------------------------------------------------------------------

    text = str(
        getattr(
            chunk,
            "enriched_content",
            "",
        )
        or ""
    )

    if not text.strip():
        return []

    mentions = _find_entity_mentions(
        text,
        entities,
    )

    for _, _, entity in mentions:

        name = str(
            getattr(entity, "name", "")
            or ""
        ).strip()

        if name:
            result.append(name)

    return sorted(
        set(result),
        key=str.lower,
    )


# ============================================================================
# Chunk entity lookup
# ============================================================================

def _chunk_entity_lookup(
    chunk: Chunk,
    entities: list[Entity],
) -> dict[str, str]:
    """
    Return:

        lowercase entity name -> canonical entity name
    """

    return {
        name.lower(): name
        for name in _entities_in_chunk(
            chunk,
            entities,
        )
    }


# ============================================================================
# Entity span
# ============================================================================

def _find_entity_span(
    text: str,
    entity_name: str,
) -> tuple[int, int] | None:
    """
    Find first explicit occurrence of an entity.
    """

    if not text or not entity_name:
        return None

    match = re.search(
        r"(?<!\w)"
        + re.escape(
            entity_name.lower()
        )
        + r"(?!\w)",
        text.lower(),
    )

    if not match:
        return None

    return (
        match.start(),
        match.end(),
    )


# ============================================================================
# Heuristic relation patterns
# ============================================================================

# IMPORTANT:
# More specific patterns must appear BEFORE generic patterns.

_HEURISTIC_PATTERNS: list[
    tuple[str, str, str]
] = [

    # ------------------------------------------------------------------
    # Reverse constructions first
    # ------------------------------------------------------------------

    (
        r"\bиспользуется\s+для\s+лечения\b",
        "TREATS",
        "reverse",
    ),

    (
        r"\bиспользуется\s+для\b",
        "USED_FOR",
        "reverse",
    ),

    (
        r"\bиспользуется\b",
        "USES",
        "reverse",
    ),

    (
        r"\bприменяется\s+для\b",
        "USED_FOR",
        "reverse",
    ),

    (
        r"\bприменяется\b",
        "USES",
        "reverse",
    ),

    (
        r"\bреализуется\b",
        "IMPLEMENTS",
        "reverse",
    ),

    (
        r"\bподдерживается\b",
        "SUPPORTS",
        "reverse",
    ),

    # ------------------------------------------------------------------
    # Russian forward
    # ------------------------------------------------------------------

    (
        r"\bиспользует\b",
        "USES",
        "forward",
    ),

    (
        r"\bиспользуют\b",
        "USES",
        "forward",
    ),

    (
        r"\bприменяет\b",
        "USES",
        "forward",
    ),

    (
        r"\bприменяют\b",
        "USES",
        "forward",
    ),

    (
        r"\bреализует\b",
        "IMPLEMENTS",
        "forward",
    ),

    (
        r"\bреализуют\b",
        "IMPLEMENTS",
        "forward",
    ),

    (
        r"\bпредоставляет\b",
        "PROVIDES",
        "forward",
    ),

    (
        r"\bобеспечивает\b",
        "PROVIDES",
        "forward",
    ),

    (
        r"\bпозволяет\b",
        "ENABLES",
        "forward",
    ),

    (
        r"\bсодержит\b",
        "CONTAINS",
        "forward",
    ),

    (
        r"\bвключает(?:\s+в\s+себя)?\b",
        "CONTAINS",
        "forward",
    ),

    (
        r"\bсостоит\s+из\b",
        "COMPOSED_OF",
        "forward",
    ),

    (
        r"\bявляется\s+частью\b",
        "PART_OF",
        "forward",
    ),

    (
        r"\bявляется\s+компонентом\b",
        "PART_OF",
        "forward",
    ),

    (
        r"\bвходит\s+в(?:\s+состав)?\b",
        "PART_OF",
        "forward",
    ),

    (
        r"\bзависит\s+от\b",
        "DEPENDS_ON",
        "forward",
    ),

    (
        r"\bоснован(?:а|о|ы)?\s+на\b",
        "BASED_ON",
        "forward",
    ),

    (
        r"\bосновывается\s+на\b",
        "BASED_ON",
        "forward",
    ),

    (
        r"\bбазируется\s+на\b",
        "BASED_ON",
        "forward",
    ),

    (
        r"\bпостроен(?:а|о|ы)?\s+на\b",
        "BUILDS_ON",
        "forward",
    ),

    (
        r"\bрасширяет\b",
        "EXTENDS",
        "forward",
    ),

    (
        r"\bулучшает\b",
        "IMPROVES",
        "forward",
    ),

    (
        r"\bповышает\b",
        "IMPROVES",
        "forward",
    ),

    (
        r"\bрешает\b",
        "SOLVES",
        "forward",
    ),

    (
        r"\bопределяет\b",
        "DEFINES",
        "forward",
    ),

    (
        r"\bизвлекает\b",
        "EXTRACTS",
        "forward",
    ),

    (
        r"\bклассифицирует\b",
        "CLASSIFIES",
        "forward",
    ),

    (
        r"\bвыбирает\b",
        "SELECTS",
        "forward",
    ),

    (
        r"\bотбирает\b",
        "SELECTS",
        "forward",
    ),

    (
        r"\bсвязывает\b",
        "CONNECTS",
        "forward",
    ),

    (
        r"\bсоединяет\b",
        "CONNECTS",
        "forward",
    ),

    (
        r"\bнаходится\s+в\b",
        "LOCATED_IN",
        "forward",
    ),

    (
        r"\bрасположен(?:а|о|ы)?\s+в\b",
        "LOCATED_IN",
        "forward",
    ),

    (
        r"\bпроизводит\b",
        "PRODUCES",
        "forward",
    ),

    (
        r"\bгенерирует\b",
        "PRODUCES",
        "forward",
    ),

    (
        r"\bтребует\b",
        "REQUIRES",
        "forward",
    ),

    (
        r"\bподдерживает\b",
        "SUPPORTS",
        "forward",
    ),

    (
        r"\bвызывает\b",
        "CAUSES",
        "forward",
    ),

    (
        r"\bприводит\s+к\b",
        "CAUSES",
        "forward",
    ),

    (
        r"\bлечит\b",
        "TREATS",
        "forward",
    ),

    (
        r"\bпредназначен(?:а|о|ы)?\s+для\b",
        "USED_FOR",
        "forward",
    ),

    (
        r"\bполучен(?:а|о|ы)?\s+из\b",
        "DERIVED_FROM",
        "forward",
    ),

    (
        r"\bпроисходит\s+из\b",
        "DERIVED_FROM",
        "forward",
    ),

    (
        r"\bпредлагает\b",
        "PROPOSES",
        "forward",
    ),

    (
        r"\bпредставляет\b",
        "INTRODUCES",
        "forward",
    ),

    (
        r"\bсравнивает\b",
        "COMPARES",
        "forward",
    ),

    # ------------------------------------------------------------------
    # English
    # ------------------------------------------------------------------

    (
        r"\buses\b",
        "USES",
        "forward",
    ),

    (
        r"\butilizes\b",
        "USES",
        "forward",
    ),

    (
        r"\bapplies\b",
        "USES",
        "forward",
    ),

    (
        r"\bimplements\b",
        "IMPLEMENTS",
        "forward",
    ),

    (
        r"\bprovides\b",
        "PROVIDES",
        "forward",
    ),

    (
        r"\benables\b",
        "ENABLES",
        "forward",
    ),

    (
        r"\ballows\b",
        "ENABLES",
        "forward",
    ),

    (
        r"\bcontains\b",
        "CONTAINS",
        "forward",
    ),

    (
        r"\bincludes\b",
        "CONTAINS",
        "forward",
    ),

    (
        r"\bconsists\s+of\b",
        "COMPOSED_OF",
        "forward",
    ),

    (
        r"\bis\s+part\s+of\b",
        "PART_OF",
        "forward",
    ),

    (
        r"\bbelongs\s+to\b",
        "PART_OF",
        "forward",
    ),

    (
        r"\bdepends\s+on\b",
        "DEPENDS_ON",
        "forward",
    ),

    (
        r"\bbased\s+on\b",
        "BASED_ON",
        "forward",
    ),

    (
        r"\bbuilt\s+on\b",
        "BUILDS_ON",
        "forward",
    ),

    (
        r"\bextends\b",
        "EXTENDS",
        "forward",
    ),

    (
        r"\bimproves\b",
        "IMPROVES",
        "forward",
    ),

    (
        r"\benhances\b",
        "IMPROVES",
        "forward",
    ),

    (
        r"\bsolves\b",
        "SOLVES",
        "forward",
    ),

    (
        r"\bdefines\b",
        "DEFINES",
        "forward",
    ),

    (
        r"\bextracts\b",
        "EXTRACTS",
        "forward",
    ),

    (
        r"\bclassifies\b",
        "CLASSIFIES",
        "forward",
    ),

    (
        r"\bselects\b",
        "SELECTS",
        "forward",
    ),

    (
        r"\bconnects\b",
        "CONNECTS",
        "forward",
    ),

    (
        r"\bconnected\s+to\b",
        "CONNECTED_TO",
        "forward",
    ),

    (
        r"\blocated\s+in\b",
        "LOCATED_IN",
        "forward",
    ),

    (
        r"\bsituated\s+in\b",
        "LOCATED_IN",
        "forward",
    ),

    (
        r"\bproduces\b",
        "PRODUCES",
        "forward",
    ),

    (
        r"\bgenerates\b",
        "PRODUCES",
        "forward",
    ),

    (
        r"\brequires\b",
        "REQUIRES",
        "forward",
    ),

    (
        r"\bsupports\b",
        "SUPPORTS",
        "forward",
    ),

    (
        r"\bcauses\b",
        "CAUSES",
        "forward",
    ),

    (
        r"\bleads\s+to\b",
        "CAUSES",
        "forward",
    ),

    (
        r"\bresults\s+in\b",
        "CAUSES",
        "forward",
    ),

    (
        r"\btreats\b",
        "TREATS",
        "forward",
    ),

    (
        r"\bused\s+for\b",
        "USED_FOR",
        "forward",
    ),

    (
        r"\bdesigned\s+for\b",
        "USED_FOR",
        "forward",
    ),

    (
        r"\bderived\s+from\b",
        "DERIVED_FROM",
        "forward",
    ),

    (
        r"\bcomes\s+from\b",
        "DERIVED_FROM",
        "forward",
    ),

    (
        r"\bintroduces\b",
        "INTRODUCES",
        "forward",
    ),

    (
        r"\bproposes\b",
        "PROPOSES",
        "forward",
    ),

    (
        r"\bcompares\b",
        "COMPARES",
        "forward",
    ),
]


# ============================================================================
# Relation extraction
# ============================================================================

def _relation_from_phrase(
    phrase: str,
) -> str:
    """
    Try to recognize a canonical relation from a phrase.
    """

    phrase = str(
        phrase or ""
    ).strip().lower()

    if not phrase:
        return ""

    if phrase in _RELATION_ALIASES:
        return _RELATION_ALIASES[phrase]

    aliases = sorted(
        _RELATION_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, relation in aliases:

        pattern = (
            r"(?<!\w)"
            + re.escape(alias)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            phrase,
        ):
            return relation

    return ""


def _extract_relation_between_entities(
    text: str,
    first_entity: str,
    second_entity: str,
    max_chars: int = DEFAULT_HEURISTIC_WINDOW,
) -> list[tuple[str, str, str]]:
    """
    Detect explicit semantic relation between two entities.

    Returns:

        relation,
        direction,
        evidence
    """

    first_span = _find_entity_span(
        text,
        first_entity,
    )

    second_span = _find_entity_span(
        text,
        second_entity,
    )

    if not first_span or not second_span:
        return []

    if first_span[0] <= second_span[0]:

        left_entity = first_entity
        right_entity = second_entity

        left_end = first_span[1]
        right_start = second_span[0]

    else:

        left_entity = second_entity
        right_entity = first_entity

        left_end = second_span[1]
        right_start = first_span[0]

    if right_start <= left_end:
        return []

    between = text[
        left_end:right_start
    ]

    if len(between) > max_chars:
        between = between[:max_chars]

    candidates: list[
        tuple[str, str, str]
    ] = []

    for pattern, relation, direction in _HEURISTIC_PATTERNS:

        match = re.search(
            pattern,
            between,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        evidence = match.group(
            0
        ).strip()

        candidates.append(
            (
                relation,
                direction,
                evidence,
            )
        )

    if not candidates:

        relation = _relation_from_phrase(
            between
        )

        if relation:
            candidates.append(
                (
                    relation,
                    "forward",
                    between.strip(),
                )
            )

    unique: list[
        tuple[str, str, str]
    ] = []

    seen: set[
        tuple[str, str]
    ] = set()

    for relation, direction, evidence in candidates:

        relation = _normalize_relation(
            relation
        )

        if not _is_valid_relation_candidate(
            relation
        ):
            continue

        key = (
            relation,
            direction,
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            (
                relation,
                direction,
                evidence,
            )
        )

    return unique


# ============================================================================
# Relationship object builder
# ============================================================================

def _build_heuristic_relationship(
    source: Entity,
    target: Entity,
    relation: str,
    description: str,
    chunk: Chunk,
    method: str = "heuristic",
) -> Relationship | None:
    """
    Construct a validated Relationship object.
    """

    relation = _normalize_relation(
        relation
    )

    if not _is_valid_relation_candidate(
        relation
    ):
        return None

    source_name = str(
        source.name or ""
    ).strip()

    target_name = str(
        target.name or ""
    ).strip()

    if not source_name or not target_name:
        return None

    if source_name.lower() == target_name.lower():
        return None

    return Relationship(
        id=_relation_id(
            source_name,
            relation,
            target_name,
        ),
        source=source_name,
        target=target_name,
        relation_type=relation,
        description=str(
            description or ""
        ).strip(),
        weight=1.0,
        metadata={
            "method": method,
            "source_chunk": str(chunk.id),
        },
    )


# ============================================================================
# Heuristic discovery
# ============================================================================

def discover_heuristic(
    chunks: list[Chunk],
    entities: list[Entity],
) -> list[Relationship]:
    """
    Discover relationships without LLM.

    Only explicit semantic patterns are accepted.
    """

    relationships: list[
        Relationship
    ] = []

    if not chunks:
        return relationships

    if not entities:

        logger.warning(
            "Heuristic discovery skipped: no entities."
        )

        return relationships

    cfg = get_settings()

    window = getattr(
        cfg.relationship_discovery,
        "context_window",
        DEFAULT_HEURISTIC_WINDOW,
    )

    try:
        window = max(
            50,
            int(window),
        )
    except Exception:
        window = DEFAULT_HEURISTIC_WINDOW

    logger.info(
        "Starting heuristic relationship discovery: "
        "chunks=%d, entities=%d, window=%d",
        len(chunks),
        len(entities),
        window,
    )

    for chunk in chunks:

        text = str(
            chunk.enriched_content or ""
        )

        if not text.strip():
            continue

        mentions = _find_entity_mentions(
            text,
            entities,
        )

        chunk_entity_names = [
            entity.name
            for _, _, entity in mentions
        ]

        logger.debug(
            "HEURISTIC chunk=%s entities=%s",
            chunk.id,
            chunk_entity_names,
        )

        if len(mentions) < 2:
            continue

        for i in range(
            len(mentions)
        ):

            _, _, entity_a = mentions[i]

            for j in range(
                i + 1,
                len(mentions),
            ):

                _, _, entity_b = mentions[j]

                if (
                    entity_a.name.lower()
                    == entity_b.name.lower()
                ):
                    continue

                detected = (
                    _extract_relation_between_entities(
                        text,
                        entity_a.name,
                        entity_b.name,
                        max_chars=window,
                    )
                )

                if not detected:
                    continue

                for (
                    relation,
                    direction,
                    evidence,
                ) in detected:

                    if direction == "forward":

                        source = entity_a
                        target = entity_b

                    else:

                        source = entity_b
                        target = entity_a

                    relationship = (
                        _build_heuristic_relationship(
                            source=source,
                            target=target,
                            relation=relation,
                            description=(
                                f"Detected from explicit "
                                f"pattern: {evidence}"
                            ),
                            chunk=chunk,
                        )
                    )

                    if relationship is None:
                        continue

                    relationships.append(
                        relationship
                    )

                    logger.debug(
                        "HEURISTIC relation: "
                        "%s --%s--> %s "
                        "(chunk=%s, evidence=%r)",
                        relationship.source,
                        relationship.relation_type,
                        relationship.target,
                        chunk.id,
                        evidence,
                    )

    logger.info(
        "Heuristic discovery produced %d candidates",
        len(relationships),
    )

    return relationships


# ============================================================================
# LLM prompt
# ============================================================================

def _build_llm_prompt(
    chunks: list[Chunk],
    entities: list[Entity],
    approved: list[str],
) -> str:
    """
    Build semantic discovery prompt.

    Each chunk receives only entities explicitly mentioned in that chunk.
    """

    approved_text = (
        ", ".join(approved)
        if approved
        else "No approved relationship schema exists yet."
    )

    sections: list[str] = []

    for index, chunk in enumerate(
        chunks,
        1,
    ):

        chunk_entities = _entities_in_chunk(
            chunk,
            entities,
        )

        content = str(
            chunk.enriched_content or ""
        )

        sections.append(
            f"""
CHUNK {index}
ID: {chunk.id}

ENTITIES IN THIS CHUNK:
{json.dumps(chunk_entities, ensure_ascii=False)}

TEXT:
{content[:5000]}
"""
        )

    return f"""
You are a semantic relationship discovery system for a knowledge graph.

Your task is to discover meaningful, reusable semantic relationships
between entities explicitly supported by the text.

The text can be Russian or English.

============================================================
CORE RULE
============================================================

This is NOT entity co-occurrence detection.

The fact that two entities occur in the same sentence does NOT prove
that they have a semantic relationship.

A relationship must be supported by an explicit statement,
grammatical construction, or unambiguous semantic evidence.

============================================================
ENTITY RULE
============================================================

For every CHUNK you may ONLY use entities listed under:

ENTITIES IN THIS CHUNK

Never:

- invent an entity;
- use an entity from another chunk;
- create a relationship between entities merely because they are
  present in the global corpus;
- normalize one entity into a different entity.

============================================================
RELATION RULES
============================================================

Every relation must:

1. Be semantically meaningful.
2. Be reusable across the corpus.
3. Contain 1-5 semantic words.
4. Use UPPER_SNAKE_CASE.
5. Be no longer than 50 characters.
6. Describe the relation, not the sentence.
7. Preserve source/target direction.

GOOD:

USES
IMPLEMENTS
PROVIDES
ENABLES
CONTAINS
PART_OF
COMPOSED_OF
DEPENDS_ON
BASED_ON
BUILDS_ON
EXTENDS
IMPROVES
SOLVES
DEFINES
EXTRACTS
CLASSIFIES
SELECTS
CONNECTS
CONNECTED_TO
LOCATED_IN
PRODUCES
REQUIRES
SUPPORTS
CAUSES
TREATS
USED_FOR
DERIVED_FROM
INTRODUCES
PROPOSES
COMPARES

BAD:

RELATED_TO
ASSOCIATED_WITH
THING
CONNECTION
STUFF
AND
OR
IS

Do not use vague relations when a more precise relation is supported.

============================================================
DO NOT COPY TEXT
============================================================

Do not copy a phrase from the source as the relation.

Example:

Text:
"Graph RAG uses Neo4j to store entities."

Correct:

Graph RAG -> USES -> Neo4j

Incorrect:

Graph RAG -> USES_NEO4J_TO_STORE -> ...

============================================================
DIRECTION
============================================================

Preserve semantic direction.

Example:

"Graph RAG uses Neo4j."

Correct:

Graph RAG -> USES -> Neo4j

Not:

Neo4j -> USES -> Graph RAG

Example:

"Neo4j is used by Graph RAG."

Correct:

Graph RAG -> USES -> Neo4j

============================================================
APPROVED RELATIONSHIPS
============================================================

{approved_text}

Reuse an approved relation when it has the same semantic meaning.

Do NOT force an approved relation if the meaning differs.

============================================================
PRECISION
============================================================

Prefer precision over quantity.

If the evidence is ambiguous, return no relationship.

Do not create a relationship simply to increase the number of results.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Format:

{{
  "relationships": [
    {{
      "source": "entity",
      "target": "entity",
      "relation": "PREDICATE",
      "description": "brief explanation"
    }}
  ]
}}

Requirements:

- source and target must be entities from the same CHUNK;
- source and target must be different;
- relation must be UPPER_SNAKE_CASE;
- relation must contain no more than 5 words;
- relation must be <= 50 characters;
- relation must be semantically reusable;
- relation must be explicitly supported;
- do not return sentence fragments;
- do not return source text as relation;
- do not invent entities;
- do not invent relationships.

If no valid relationships exist:

{{
  "relationships": []
}}

DOCUMENTS:

{"".join(sections)}
"""


# ============================================================================
# LLM response parser
# ============================================================================

def _parse_llm_relationships(
    response_text: str,
    chunks: list[Chunk],
    entities: list[Entity],
) -> list[Relationship]:
    """
    Parse and strictly validate LLM output.

    Every source/target pair must occur in the SAME chunk.
    """

    text = str(
        response_text or ""
    ).strip()

    if not text:
        return []

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
    ).strip()

    try:

        data = json.loads(text)

    except json.JSONDecodeError as exc:

        logger.warning(
            "LLM relationship discovery returned invalid JSON: %s",
            exc,
        )

        return []

    if not isinstance(
        data,
        dict,
    ):
        return []

    raw_relationships = data.get(
        "relationships",
        [],
    )

    if not isinstance(
        raw_relationships,
        list,
    ):
        return []

    # ------------------------------------------------------------------
    # Chunk-specific entity lookup
    # ------------------------------------------------------------------

    chunk_entities: dict[
        str,
        dict[str, str],
    ] = {}

    for chunk in chunks:

        chunk_entities[
            str(chunk.id)
        ] = _chunk_entity_lookup(
            chunk,
            entities,
        )

    # ------------------------------------------------------------------
    # Global canonical entity lookup
    # ------------------------------------------------------------------

    canonical_entities = {
        str(entity.name).strip().lower():
        str(entity.name).strip()
        for entity in entities
        if str(entity.name).strip()
    }

    results: list[
        Relationship
    ] = []

    seen_ids: set[str] = set()

    for item in raw_relationships:

        if not isinstance(
            item,
            dict,
        ):
            continue

        source_raw = str(
            item.get(
                "source",
                "",
            )
        ).strip()

        target_raw = str(
            item.get(
                "target",
                "",
            )
        ).strip()

        relation_raw = str(
            item.get(
                "relation",
                "",
            )
        ).strip()

        description = str(
            item.get(
                "description",
                "",
            )
        ).strip()

        if not source_raw or not target_raw:
            continue

        if (
            source_raw.lower()
            == target_raw.lower()
        ):
            continue

        source_canonical = canonical_entities.get(
            source_raw.lower()
        )

        target_canonical = canonical_entities.get(
            target_raw.lower()
        )

        if not source_canonical:

            logger.debug(
                "Rejected LLM relation: "
                "unknown source entity %r",
                source_raw,
            )

            continue

        if not target_canonical:

            logger.debug(
                "Rejected LLM relation: "
                "unknown target entity %r",
                target_raw,
            )

            continue

        # --------------------------------------------------------------
        # Same-chunk validation
        # --------------------------------------------------------------

        valid_chunk: Chunk | None = None

        for chunk in chunks:

            lookup = chunk_entities.get(
                str(chunk.id),
                {},
            )

            if (
                source_canonical.lower() in lookup
                and target_canonical.lower() in lookup
            ):

                valid_chunk = chunk
                break

        if valid_chunk is None:

            logger.debug(
                "Rejected cross-chunk/nonexistent pair: "
                "%s -> %s",
                source_raw,
                target_raw,
            )

            continue

        # --------------------------------------------------------------
        # Relation validation
        # --------------------------------------------------------------

        relation = _normalize_relation(
            relation_raw
        )

        if not relation:
            continue

        if not _is_valid_relation_candidate(
            relation
        ):

            logger.debug(
                "Rejected invalid LLM relation: %r",
                relation_raw,
            )

            continue

        relationship_id = _relation_id(
            source_canonical,
            relation,
            target_canonical,
        )

        if relationship_id in seen_ids:
            continue

        seen_ids.add(
            relationship_id
        )

        relationship = Relationship(
            id=relationship_id,
            source=source_canonical,
            target=target_canonical,
            relation_type=relation,
            description=description,
            weight=1.0,
            metadata={
                "method": "llm",
                "source_chunk": str(
                    valid_chunk.id
                ),
            },
        )

        results.append(
            relationship
        )

    logger.info(
        "LLM parser accepted %d/%d candidates",
        len(results),
        len(raw_relationships),
    )

    return results


# ============================================================================
# LLM discovery
# ============================================================================

def discover_llm(
    chunks: list[Chunk],
    entities: list[Entity],
    openai_client: Any | None = None,
) -> list[Relationship]:
    """
    LLM semantic relationship discovery.

    Several chunks are processed in one request.
    """

    if not chunks:
        return []

    if not entities:

        logger.warning(
            "LLM discovery skipped: no entities available."
        )

        return []

    cfg = get_settings()

    if openai_client is None:

        from rag_core.config import make_openai_client

        openai_client = make_openai_client(
            cfg
        )

    approved = (
        approved_relation_names()
        if cfg.relationship_discovery.use_approved_schema
        else []
    )

    batch_size = max(
        1,
        int(
            cfg.relationship_discovery.batch_size
        ),
    )

    max_per_chunk = max(
        1,
        int(
            cfg.relationship_discovery
            .max_relationships_per_chunk
        ),
    )

    results: list[
        Relationship
    ] = []

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):

        batch = chunks[
            start:start + batch_size
        ]

        useful_batch: list[Chunk] = []

        for chunk in batch:

            chunk_entities = _entities_in_chunk(
                chunk,
                entities,
            )

            if len(chunk_entities) >= 2:
                useful_batch.append(
                    chunk
                )

        if not useful_batch:

            logger.debug(
                "Skipping LLM batch %d-%d: "
                "no chunks contain >=2 entities",
                start + 1,
                min(
                    start + batch_size,
                    len(chunks),
                ),
            )

            continue

        prompt = _build_llm_prompt(
            useful_batch,
            entities,
            approved,
        )

        try:

            response = (
                openai_client.chat.completions.create(
                    model=(
                        cfg.openai
                        .relationship_discovery_model
                    ),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                cfg.openai
                                .relationship_discovery_system_prompt
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.0,
                )
            )

            text = (
                response.choices[0]
                .message.content
                or ""
            )

            batch_relationships = (
                _parse_llm_relationships(
                    text,
                    useful_batch,
                    entities,
                )
            )

            per_chunk_counter: Counter = Counter()

            filtered_batch: list[
                Relationship
            ] = []

            for relationship in batch_relationships:

                chunk_id = str(
                    relationship.metadata.get(
                        "source_chunk",
                        "",
                    )
                )

                if (
                    per_chunk_counter[
                        chunk_id
                    ]
                    >= max_per_chunk
                ):
                    continue

                per_chunk_counter[
                    chunk_id
                ] += 1

                filtered_batch.append(
                    relationship
                )

            results.extend(
                filtered_batch
            )

            logger.info(
                "LLM discovery batch %d-%d: "
                "%d candidates",
                start + 1,
                min(
                    start + batch_size,
                    len(chunks),
                ),
                len(filtered_batch),
            )

        except Exception as exc:

            logger.exception(
                "LLM relationship discovery failed: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # Global deduplication
    # ------------------------------------------------------------------

    unique: dict[
        str,
        Relationship,
    ] = {}

    for relationship in results:

        unique[
            relationship.id
        ] = relationship

    results = list(
        unique.values()
    )

    logger.info(
        "LLM discovery completed: %d unique relationships",
        len(results),
    )

    return results


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_relationships(
    relationships: list[Relationship],
) -> list[dict[str, Any]]:
    """
    Aggregate candidate relationships by relation type.

    Frequency:
        number of occurrences.

    Unique pairs:
        number of unique source-target pairs.
    """

    stats: dict[
        str,
        dict[str, Any],
    ] = {}

    pair_examples: defaultdict[
        str,
        dict[
            tuple[str, str],
            dict[str, str],
        ],
    ] = defaultdict(dict)

    methods: defaultdict[
        str,
        Counter,
    ] = defaultdict(Counter)

    for rel in relationships:

        relation = _normalize_relation(
            rel.relation_type
        )

        if not _is_valid_relation_candidate(
            relation
        ):
            continue

        source = str(
            rel.source or ""
        ).strip()

        target = str(
            rel.target or ""
        ).strip()

        if not source or not target:
            continue

        if (
            source.lower()
            == target.lower()
        ):
            continue

        if relation not in stats:

            stats[relation] = {
                "relation": relation,
                "frequency": 0,
                "unique_pairs": set(),
            }

        item = stats[
            relation
        ]

        item[
            "frequency"
        ] += 1

        pair = (
            source.lower(),
            target.lower(),
        )

        item[
            "unique_pairs"
        ].add(pair)

        method = str(
            rel.metadata.get(
                "method",
                "unknown",
            )
        )

        methods[
            relation
        ][
            method
        ] += 1

        if pair not in pair_examples[
            relation
        ]:

            pair_examples[
                relation
            ][pair] = {
                "source": source,
                "target": target,
                "description": str(
                    rel.description or ""
                ).strip(),
            }

    result: list[
        dict[str, Any]
    ] = []

    for relation, item in stats.items():

        examples = list(
            pair_examples[
                relation
            ].values()
        )

        result.append(
            {
                "relation": relation,
                "frequency": item[
                    "frequency"
                ],
                "unique_pairs": len(
                    item[
                        "unique_pairs"
                    ]
                ),
                "methods": dict(
                    methods[
                        relation
                    ]
                ),
                "examples": examples[:5],
                "pairs": examples,
            }
        )

    result.sort(
        key=lambda x: (
            -x["frequency"],
            -x["unique_pairs"],
            x["relation"],
        )
    )

    return result


# ============================================================================
# Main discovery pipeline
# ============================================================================

def discover_relationships(
    chunks: list[Chunk],
    entities: list[Entity],
    openai_client: Any | None = None,
) -> dict[str, Any]:
    """
    Run configured relationship discovery.

    Returns:

        {
            "mode": "...",
            "raw_relationships": [...],
            "candidates": [...]
        }
    """

    cfg = get_settings()

    if not cfg.relationship_discovery.enabled:

        return {
            "mode": "disabled",
            "raw_relationships": [],
            "candidates": [],
        }

    mode = str(
        cfg.relationship_discovery.mode
    ).strip().lower()

    if mode not in {
        "heuristic",
        "llm",
        "both",
    }:

        raise ValueError(
            "RELATIONSHIP_DISCOVERY_MODE must be "
            "'heuristic', 'llm' or 'both'."
        )

    if not chunks:

        return {
            "mode": mode,
            "raw_relationships": [],
            "candidates": [],
        }

    if not entities:

        logger.warning(
            "Relationship discovery: no entities."
        )

        return {
            "mode": mode,
            "raw_relationships": [],
            "candidates": [],
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    max_documents = max(
        1,
        int(
            cfg.relationship_discovery
            .max_documents
        ),
    )

    max_chunks_per_document = max(
        1,
        int(
            cfg.relationship_discovery
            .max_chunks_per_document
        ),
    )

    max_selected_chunks = (
        max_documents
        * max_chunks_per_document
    )

    selected_chunks = chunks[
        :max_selected_chunks
    ]

    logger.info(
        "Relationship discovery: "
        "mode=%s selected_chunks=%d entities=%d",
        mode,
        len(selected_chunks),
        len(entities),
    )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    raw: list[
        Relationship
    ] = []

    if mode in {
        "heuristic",
        "both",
    }:

        heuristic_results = discover_heuristic(
            selected_chunks,
            entities,
        )

        logger.info(
            "DISCOVERY HEURISTIC RESULT: %d",
            len(heuristic_results),
        )

        raw.extend(
            heuristic_results
        )

    if mode in {
        "llm",
        "both",
    }:

        llm_results = discover_llm(
            selected_chunks,
            entities,
            openai_client,
        )

        logger.info(
            "DISCOVERY LLM RESULT: %d",
            len(llm_results),
        )

        raw.extend(
            llm_results
        )

    logger.info(
        "DISCOVERY RAW RESULT: %d",
        len(raw),
    )

    # ------------------------------------------------------------------
    # Build chunk/entity lookup
    # ------------------------------------------------------------------

    chunk_entity_lookup: dict[
        str,
        set[str],
    ] = {}

    for chunk in selected_chunks:

        chunk_entity_lookup[
            str(chunk.id)
        ] = {
            name.lower()
            for name in _entities_in_chunk(
                chunk,
                entities,
            )
        }

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------

    validated: list[
        Relationship
    ] = []

    for rel in raw:

        source = str(
            rel.source or ""
        ).strip()

        target = str(
            rel.target or ""
        ).strip()

        if not source or not target:
            continue

        if (
            source.lower()
            == target.lower()
        ):
            continue

        relation = _normalize_relation(
            rel.relation_type
        )

        if not _is_valid_relation_candidate(
            relation
        ):
            continue

        chunk_id = str(
            rel.metadata.get(
                "source_chunk",
                "",
            )
        )

        # --------------------------------------------------------------
        # If source_chunk exists, BOTH entities must occur in that chunk.
        # --------------------------------------------------------------

        if chunk_id:

            chunk_entities = (
                chunk_entity_lookup.get(
                    chunk_id,
                    set(),
                )
            )

            source_in_chunk = (
                source.lower()
                in chunk_entities
            )

            target_in_chunk = (
                target.lower()
                in chunk_entities
            )

            if not (
                source_in_chunk
                and target_in_chunk
            ):

                logger.debug(
                    "Dropped relation because "
                    "both endpoints are not in the same chunk: "
                    "%s --%s--> %s chunk=%s",
                    source,
                    relation,
                    target,
                    chunk_id,
                )

                continue

        rel.relation_type = relation

        validated.append(
            rel
        )

    logger.info(
        "DISCOVERY VALIDATED RESULT: %d",
        len(validated),
    )

    # ------------------------------------------------------------------
    # Keep all occurrences for frequency aggregation.
    # ------------------------------------------------------------------

    raw = validated

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    logger.info(
        "DISCOVERY BEFORE AGGREGATION: %d",
        len(raw),
    )

    candidates = aggregate_relationships(
        raw
    )

    # ------------------------------------------------------------------
    # Frequency filter
    # ------------------------------------------------------------------

    min_frequency = max(
        1,
        int(
            cfg.relationship_discovery
            .min_frequency
        ),
    )

    candidates = [
        candidate
        for candidate in candidates
        if candidate["frequency"]
        >= min_frequency
    ]

    # ------------------------------------------------------------------
    # Max candidates
    # ------------------------------------------------------------------

    max_candidates = max(
        1,
        int(
            cfg.relationship_discovery
            .max_candidates
        ),
    )

    candidates = candidates[
        :max_candidates
    ]

    # ------------------------------------------------------------------
    # JSON-safe raw relationships
    # ------------------------------------------------------------------

    raw_relationships: list[
        dict[str, Any]
    ] = []

    for rel in raw:

        raw_relationships.append(
            {
                "id": rel.id,
                "source": rel.source,
                "target": rel.target,
                "relation": rel.relation_type,
                "description": rel.description,
                "weight": rel.weight,
                "metadata": dict(
                    rel.metadata or {}
                ),
            }
        )

    logger.info(
        "Relationship discovery completed: "
        "mode=%s raw=%d candidates=%d",
        mode,
        len(raw_relationships),
        len(candidates),
    )

    return {
        "mode": mode,
        "raw_relationships": raw_relationships,
        "candidates": candidates,
        "threshold": min_frequency,
    }


# ============================================================================
# Human-approved schema
# ============================================================================

def save_approved_schema(
    approved_candidates: list[dict[str, Any]],
    driver: Any | None = None,
) -> Path:
    """
    Save human-approved relationship types.

    Also persists selected source-target pairs to Neo4j.
    """

    cfg = get_settings()

    path = Path(
        cfg.relationship_discovery.schema_path
    )

    if not path.is_absolute():
        path = Path.cwd() / path

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    relationships: list[
        dict[str, Any]
    ] = []

    for candidate in approved_candidates:

        if not isinstance(
            candidate,
            dict,
        ):
            continue

        name = _normalize_relation(
            str(
                candidate.get(
                    "relation",
                    "",
                )
            )
        )

        if not _is_valid_relation_candidate(
            name
        ):
            continue

        description = str(
            candidate.get(
                "description",
                "",
            )
        ).strip()

        examples = candidate.get(
            "examples",
            [],
        )

        if (
            not description
            and isinstance(
                examples,
                list,
            )
            and examples
        ):

            first = examples[0]

            if isinstance(
                first,
                dict,
            ):

                description = str(
                    first.get(
                        "description",
                        "",
                    )
                ).strip()

        relationships.append(
            {
                "name": name,
                "description": description,
                "frequency": candidate.get(
                    "frequency",
                    0,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Deduplicate relation names
    # ------------------------------------------------------------------

    unique_relationships: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in relationships:

        unique_relationships[
            item["name"]
        ] = item

    relationships = list(
        unique_relationships.values()
    )

    data = {
        "version": 1,
        "relationships": relationships,
    }

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Saved %d approved relationship types to %s",
        len(relationships),
        path,
    )

    # ------------------------------------------------------------------
    # No Neo4j driver
    # ------------------------------------------------------------------

    if driver is None:

        logger.info(
            "Neo4j driver not provided; "
            "approved schema saved only."
        )

        return path

    # ------------------------------------------------------------------
    # Build selected edges
    # ------------------------------------------------------------------

    db_relationships: list[
        Relationship
    ] = []

    for candidate in approved_candidates:

        if not isinstance(
            candidate,
            dict,
        ):
            continue

        relation_type = _normalize_relation(
            str(
                candidate.get(
                    "relation",
                    "",
                )
            )
        )

        if not _is_valid_relation_candidate(
            relation_type
        ):
            continue

        pairs = candidate.get(
            "pairs",
            [],
        )

        if not isinstance(
            pairs,
            list,
        ):
            continue

        for pair in pairs:

            if not isinstance(
                pair,
                dict,
            ):
                continue

            source = str(
                pair.get(
                    "source",
                    "",
                )
            ).strip()

            target = str(
                pair.get(
                    "target",
                    "",
                )
            ).strip()

            description = str(
                pair.get(
                    "description",
                    "",
                )
            ).strip()

            if not source or not target:
                continue

            if (
                source.lower()
                == target.lower()
            ):
                continue

            db_relationships.append(
                Relationship(
                    id=_relation_id(
                        source,
                        relation_type,
                        target,
                    ),
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    description=description,
                    weight=1.0,
                    metadata={
                        "method": (
                            "approved_discovery"
                        ),
                    },
                )
            )

    # ------------------------------------------------------------------
    # Deduplicate edges
    # ------------------------------------------------------------------

    unique_edges: dict[
        str,
        Relationship,
    ] = {}

    for rel in db_relationships:

        unique_edges[
            rel.id
        ] = rel

    db_relationships = list(
        unique_edges.values()
    )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    if db_relationships:

        created_count = (
            create_phrase_relationships(
                db_relationships,
                driver,
            )
        )

        logger.info(
            "Persisted %d approved relationship edges to Neo4j",
            created_count,
        )

    else:

        logger.info(
            "No relationship pairs selected "
            "for Neo4j persistence."
        )

    return path