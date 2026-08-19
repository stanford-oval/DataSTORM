"""
Utility functions for staged analytical report generation from a Co-STORM run directory.

Pipeline:
1) Load topic + thesis + tree-selected evidence (+ warmstart context)
2) Create per-evidence contribution notes against topic/thesis
3) Generate publication-style title
4) Build high-level report plan
5) Draft each section with tree evidence as core spine, optionally enriched with web retrieval
6) Final polish pass and write report + provenance artifacts
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from knowledge_storm.langfuse_llm import get_llm, call_llm_with_structured_output


DB_DESCRIPTION_MAP = {
    "acled": (
        "You have access to an ACLED database. Armed Conflict Location & Event Data "
        "(ACLED) is a non-profit organization specializing in disaggregated conflict "
        "data collection, analysis, and crisis mapping. ACLED codes the dates, actors, "
        "locations, fatalities, and types of all reported political violence and "
        "demonstration events around the world in real time. "
        "We have data up to and until end of 2024."
    ),
    "fec": "You have access to an FEC database storing campaign finance data.",
    "sf_311": (
        "You have access to a 311 database storing service requests from the City "
        "of San Francisco."
    ),
}

DEFAULT_SERPER_PARAMS = {"autocorrect": True, "num": 10, "page": 1}


@dataclass
class SourceRecord:
    citation_id: int
    source_type: str
    url: str
    title: str
    description: str
    snippets: List[str]
    origin: str
    meta: Dict[str, Any]


@dataclass
class EvidenceRecord:
    evidence_id: int
    citation_id: int
    source_type: str
    question: str
    finding: str
    url: str
    node_id: str
    depth: int


@dataclass
class WarmstartPair:
    question: str
    answer: str


class EvidenceContributionNote(BaseModel):
    evidence_id: int
    stance: str
    contribution_summary: str
    how_it_supports_or_refutes: str
    key_angles_for_report: List[str] = Field(default_factory=list)
    caveats_or_limits: List[str] = Field(default_factory=list)


class ReportTitleOutput(BaseModel):
    title: str
    subtitle: str
    editorial_angle: str


class SectionPlanItem(BaseModel):
    section_id: str
    heading: str
    purpose: str
    must_include_evidence_ids: List[int] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    storytelling_moves: List[str] = Field(default_factory=list)
    web_queries: List[str] = Field(default_factory=list)


class ReportPlanOutput(BaseModel):
    lede_strategy: str
    key_findings: List[str] = Field(default_factory=list)
    sections: List[SectionPlanItem] = Field(default_factory=list)
    closing_strategy: str


class SectionDraftOutput(BaseModel):
    section_id: str
    heading: str
    section_markdown: str
    used_citations: List[int] = Field(default_factory=list)


class FinalReportOutput(BaseModel):
    report_markdown: str


class InferredThesisOutput(BaseModel):
    thesis: str


class SourceRegistry:
    def __init__(self) -> None:
        self._next_id = 1
        self._key_to_id: Dict[str, int] = {}
        self._records: Dict[int, SourceRecord] = {}

    def _make_key(
        self,
        source_type: str,
        url: str,
        title: str,
        description: str,
        snippets: List[str],
        origin: str,
    ) -> str:
        if url.strip():
            return f"url::{url.strip().lower()}"
        payload = "|".join(
            [
                source_type.strip().lower(),
                origin.strip().lower(),
                title.strip().lower(),
                description.strip().lower(),
                " ".join((snippets or [])[:2]).strip().lower(),
            ]
        )
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return f"hash::{digest}"

    def register(
        self,
        *,
        source_type: str,
        url: str,
        title: str,
        description: str,
        snippets: Optional[List[str]],
        origin: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        snippets = snippets or []
        key = self._make_key(
            source_type=source_type,
            url=url,
            title=title,
            description=description,
            snippets=snippets,
            origin=origin,
        )
        if key in self._key_to_id:
            return self._key_to_id[key]

        citation_id = self._next_id
        self._next_id += 1
        self._key_to_id[key] = citation_id
        self._records[citation_id] = SourceRecord(
            citation_id=citation_id,
            source_type=source_type,
            url=url,
            title=title,
            description=description,
            snippets=snippets,
            origin=origin,
            meta=meta or {},
        )
        return citation_id

    def get(self, citation_id: int) -> SourceRecord:
        return self._records[citation_id]

    def all_records(self) -> List[SourceRecord]:
        return [self._records[k] for k in sorted(self._records)]

    def available_ids(self) -> Set[int]:
        return set(self._records.keys())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def collect_tree_evidence(tree: Dict[str, Any], registry: SourceRegistry) -> List[EvidenceRecord]:
    collected: List[EvidenceRecord] = []

    def walk(node: Dict[str, Any], parent_question: str = "") -> None:
        if node.get("is_final_selected"):
            dlg = node.get("dlg_turn") or {}
            question = (dlg.get("user_utterance") or "").strip()
            summary = (dlg.get("summary") or dlg.get("agent_utterance") or "").strip()
            search_results = dlg.get("search_results") or []

            url = ""
            source_type = "database"
            title = ""
            description = ""
            snippets: List[str] = []
            if search_results:
                first_result = search_results[0] or {}
                url = first_result.get("url") or ""
                title = first_result.get("title") or ""
                description = first_result.get("description") or ""
                snippets = first_result.get("snippets") or []
                source_type = (
                    (first_result.get("meta") or {}).get("source_type")
                    or source_type
                )

            if not summary:
                summary = description or (snippets[0] if snippets else "")
            if source_type == "database" and parent_question and question and question != parent_question:
                question = f"{parent_question}\n{question}"

            if summary:
                citation_id = registry.register(
                    source_type=source_type,
                    url=url,
                    title=title or f"Tree evidence node {node.get('node_id', '')}",
                    description=description,
                    snippets=snippets,
                    origin="tree",
                    meta={
                        "node_id": node.get("node_id"),
                        "depth": node.get("depth"),
                        "question": question,
                    },
                )
                collected.append(
                    EvidenceRecord(
                        evidence_id=len(collected) + 1,
                        citation_id=citation_id,
                        source_type=source_type,
                        question=question,
                        finding=summary,
                        url=url,
                        node_id=str(node.get("node_id") or ""),
                        depth=int(node.get("depth") or 0),
                    )
                )

        this_question = ((node.get("dlg_turn") or {}).get("user_utterance") or "").strip()
        next_parent = this_question or parent_question
        for child in (node.get("children") or []):
            walk(child, parent_question=next_parent)

    walk(tree, parent_question="")
    return collected


def load_warmstart_evidence_from_disk(
    warmstart_data: List[Dict[str, Any]],
    registry: SourceRegistry,
    evidence_id_start: int = 1,
) -> List[EvidenceRecord]:
    """Reconstruct warmstart EvidenceRecords from serialized warmstart_conversation.json.

    Only processes turns that have cited_info serialized (requires ConvTurn.to_dict() to have
    saved cited_info — runs prior to that fix will have cited_info=null and produce no records).
    Mirrors the warmstart evidence registration logic in the live pipeline.
    """
    records: List[EvidenceRecord] = []
    pending_question = ""

    for turn in (warmstart_data or []):
        utterance_type = (turn.get("utterance_type") or "").lower()
        utterance = (turn.get("utterance") or turn.get("raw_utterance") or "").strip()
        if not utterance:
            continue

        if "question" in utterance_type:
            pending_question = utterance
            continue

        summary = utterance
        question = pending_question
        pending_question = ""

        cited_info_raw = turn.get("cited_info")
        if not cited_info_raw:
            continue  # old run or turn with no sources — skip

        cited_items = cited_info_raw.items() if isinstance(cited_info_raw, dict) else enumerate(cited_info_raw)
        urls: List[str] = []
        snippets: List[str] = []
        for _, info_dict in cited_items:
            url = (info_dict.get("url") or "").strip()
            if url:
                urls.append(url)
            desc = (info_dict.get("description") or "").strip()
            if desc:
                snippets.append(desc)

        url = urls[0] if urls else ""
        evidence_id = evidence_id_start + len(records)
        citation_id = registry.register(
            source_type="web_search",
            url=url,
            title=question,
            description=summary,
            snippets=snippets,
            origin="warmstart",
            meta={"question": question},
        )
        records.append(EvidenceRecord(
            evidence_id=evidence_id,
            citation_id=citation_id,
            source_type="web_search",
            question=question,
            finding=summary,
            url=url,
            node_id="",
            depth=0,
        ))

    return records


def parse_warmstart_pairs(warmstart_data: List[Dict[str, Any]]) -> List[WarmstartPair]:
    pairs: List[WarmstartPair] = []
    pending_question = ""
    for turn in warmstart_data:
        utterance_type = (turn.get("utterance_type") or "").lower()
        utterance = (turn.get("utterance") or turn.get("raw_utterance") or "").strip()
        if not utterance:
            continue

        if "question" in utterance_type:
            pending_question = utterance
            continue

        if "answer" in utterance_type or "potential answer" in utterance_type:
            pairs.append(
                WarmstartPair(
                    question=pending_question,
                    answer=utterance,
                )
            )
            pending_question = ""
    return pairs


def separate_inline_citations(text: str) -> str:
    """
    Expand grouped inline citations into individual bracketed citations.
    Examples:
    - [57,58] -> [57][58]
    - [22, 50, 55][6; 16] -> [22][50][55][6][16]
    """
    if not text:
        return ""

    def _replace(m: re.Match) -> str:
        raw = m.group(1).strip()
        parts = [p.strip() for p in re.split(r"[;,]", raw)]
        if not parts or any(not p or not p.isdigit() for p in parts):
            return m.group(0)
        return "".join(f"[{int(p)}]" for p in parts)

    return re.sub(r"\[([^\[\]]+)\]", _replace, text)


def find_citation_ids(text: str) -> Set[int]:
    normalized = separate_inline_citations(text or "")
    ids: Set[int] = set()
    for match in re.findall(r"\[(\d+)\]", normalized):
        try:
            ids.add(int(match))
        except ValueError:
            continue
    return ids


def remove_invalid_citations(text: str, allowed: Set[int]) -> str:
    normalized = separate_inline_citations(text or "")

    def _replace(m: re.Match) -> str:
        cid = int(m.group(1))
        return m.group(0) if cid in allowed else ""

    return re.sub(r"\[(\d+)\]", _replace, normalized)


def reindex_citations_by_appearance(text: str) -> Tuple[str, Dict[int, int]]:
    """
    Reindex citations by first appearance order in text, starting at 1.
    Returns (reindexed_text, old_to_new_map).
    """
    normalized = separate_inline_citations(text or "")
    old_to_new: Dict[int, int] = {}

    def _replace(m: re.Match) -> str:
        old = int(m.group(1))
        if old not in old_to_new:
            old_to_new[old] = len(old_to_new) + 1
        return f"[{old_to_new[old]}]"

    return re.sub(r"\[(\d+)\]", _replace, normalized), old_to_new


def build_core_evidence_packet(
    *,
    section: SectionPlanItem,
    evidence_records: List[EvidenceRecord],
    note_map: Dict[int, EvidenceContributionNote],
) -> Tuple[str, List[int]]:
    lines: List[str] = []
    citation_ids: List[int] = []
    evidence_by_id = {e.evidence_id: e for e in evidence_records}

    for evidence_id in section.must_include_evidence_ids:
        e = evidence_by_id.get(evidence_id)
        if not e:
            continue
        note = note_map.get(evidence_id)
        citation_ids.append(e.citation_id)
        lines.append(
            "\n".join(
                [
                    f"[{e.citation_id}] CORE TREE EVIDENCE",
                    f"question: {e.question}",
                    f"finding: {e.finding}",
                    f"contribution: {note.contribution_summary if note else ''}",
                    f"stance: {note.stance if note else 'unknown'}",
                ]
            )
        )
    return "\n\n".join(lines), sorted(set(citation_ids))


def build_web_packet(citation_ids: List[int], registry: SourceRegistry) -> str:
    if not citation_ids:
        return "No supplemental web sources."
    lines: List[str] = []
    for cid in citation_ids:
        src = registry.get(cid)
        snippet = src.description or (src.snippets[0] if src.snippets else "")
        lines.append(
            "\n".join(
                [
                    f"[{cid}] WEB CONTEXT",
                    f"title: {src.title}",
                    f"url: {src.url}",
                    f"snippet: {snippet}",
                ]
            )
        )
    return "\n\n".join(lines)


def build_pre_final_draft(
    *,
    thesis: str,
    title: ReportTitleOutput,
    plan: ReportPlanOutput,
    sections: List[SectionDraftOutput],
) -> str:
    lines: List[str] = [f"# {title.title}"]
    if title.subtitle.strip():
        lines.append(f"*{title.subtitle.strip()}*")
    lines.append("")
    lines.append("## Thesis")
    lines.append(
        thesis.strip()
        if thesis.strip()
        else "No predefined thesis; this report presents an evidence-based analysis of the topic."
    )
    lines.append("")
    lines.append("## Key Findings")
    if plan.key_findings:
        for finding in plan.key_findings:
            lines.append(f"- {finding}")
    else:
        lines.append("- Key findings will be refined in the final pass.")
    lines.append("")

    section_map = {s.section_id: s for s in sections}
    for section in plan.sections:
        drafted = section_map.get(section.section_id)
        if drafted and drafted.section_markdown.strip():
            lines.append(f"## {section.heading}")
            lines.append(drafted.section_markdown.strip())
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def strip_llm_reference_sections(text: str) -> str:
    """
    Remove any LLM-generated reference/sources section.
    Sources are appended deterministically in post-processing only.
    """
    lines = (text or "").splitlines()
    heading_pattern = re.compile(
        r"^\s{0,3}#{1,6}\s*(sources|references|bibliography|citations)\s*$",
        re.IGNORECASE,
    )
    start_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if heading_pattern.match(line.strip()):
            start_idx = i
            break
    if start_idx is None:
        return text
    kept = lines[:start_idx]
    return "\n".join(kept).rstrip() + "\n"


def build_sources_appendix(
    used_citations: Set[int],
    registry: SourceRegistry,
    citation_source_map: Optional[Dict[int, int]] = None,
) -> str:
    if not used_citations:
        return ""
    lines = ["", "---", "", "## Sources"]
    for cid in sorted(used_citations):
        source_cid = citation_source_map.get(cid, cid) if citation_source_map else cid
        src = registry.get(source_cid)
        title = src.title or "Untitled"
        stype = src.source_type or "unknown"
        if src.url:
            lines.append(f"[{cid}] {title} ({stype}) — {src.url}")
        else:
            lines.append(f"[{cid}] {title} ({stype})")
    return "\n".join(lines)


def init_serper_rm(serper_query_params: Optional[Dict[str, Any]]) -> Optional[Any]:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return None
    try:
        from knowledge_storm.rm import SerperRM  # pylint: disable=import-outside-toplevel
    except Exception as e:
        print(f"[web] unable to import SerperRM: {e}")
        return None
    params = dict(DEFAULT_SERPER_PARAMS)
    if serper_query_params:
        params.update(serper_query_params)
    return SerperRM(serper_search_api_key=api_key, k=10, query_params=params)


def retrieve_web_context_for_section(
    *,
    section: SectionPlanItem,
    serper_rm: Optional[Any],
    registry: SourceRegistry,
) -> List[int]:
    if serper_rm is None:
        return []

    collected: List[int] = []
    seen_ids: Set[int] = set()
    for query in section.web_queries:
        try:
            results = serper_rm.forward(query_or_queries=query, exclude_urls=[])
        except Exception as e:
            print(f"[web] section {section.section_id} query failed: {e}")
            continue

        for result in (results or []):
            citation_id = registry.register(
                source_type="web_search",
                url=result.get("url", ""),
                title=result.get("title", ""),
                description=result.get("description", ""),
                snippets=result.get("snippets", []) or [],
                origin="section_web_enrichment",
                meta={"section_id": section.section_id, "query": query},
            )
            if citation_id not in seen_ids:
                seen_ids.add(citation_id)
                collected.append(citation_id)
    return collected


def parse_serper_query_params(args_value: Optional[str], metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if args_value:
        try:
            return json.loads(args_value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid serper_query_params JSON: {e}") from e
    metadata_value = metadata.get("serper_query_params")
    if isinstance(metadata_value, dict):
        return metadata_value
    return None


# ---------------------------------------------------------------------------
# Langfuse-backed LLM call wrapper
# ---------------------------------------------------------------------------

def _langfuse_call(
    prompt_name: str,
    variables: dict,
    output_model: type[BaseModel],
    model_name: str,
    langfuse_readonly: bool = False,
    max_retries: int = 3,
) -> BaseModel:
    import time
    llm = get_llm(model_name=model_name, temperature=1.0)
    for attempt in range(max_retries):
        result = asyncio.run(
            call_llm_with_structured_output(
                prompt_name, variables, output_model, llm,
                langfuse_readonly=langfuse_readonly,
            )
        )
        if result is not None:
            return result
        if attempt < max_retries - 1:
            wait = 2 ** attempt
            print(f"  [retry] '{prompt_name}' attempt {attempt + 1} failed, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Langfuse call failed for prompt '{prompt_name}' after {max_retries} attempts.")


# ---------------------------------------------------------------------------
# LLM-calling pipeline functions
# ---------------------------------------------------------------------------

def generate_evidence_note(
    *,
    topic: str,
    thesis: str,
    evidence: EvidenceRecord,
    model: str,
    langfuse_readonly: bool = False,
) -> EvidenceContributionNote:
    out = _langfuse_call(
        "staged_report_evidence_note",
        {
            "topic": topic,
            "thesis": thesis,
            "evidence_id": evidence.evidence_id,
            "source_type": evidence.source_type,
            "citation_id": evidence.citation_id,
            "question": evidence.question,
            "finding": evidence.finding,
        },
        EvidenceContributionNote,
        model,
        langfuse_readonly,
    )
    out.evidence_id = evidence.evidence_id
    return out


def generate_title(
    *,
    topic: str,
    thesis: str,
    notes: List[EvidenceContributionNote],
    model: str,
    langfuse_readonly: bool = False,
) -> ReportTitleOutput:
    notes_text = []
    for note in notes:
        notes_text.append(
            f"- Evidence {note.evidence_id} ({note.stance}): {note.contribution_summary}"
        )
    digest = "\n".join(notes_text) if notes_text else "No notes available."

    return _langfuse_call(
        "staged_report_title",
        {
            "topic": topic,
            "thesis": thesis,
            "digest": digest,
        },
        ReportTitleOutput,
        model,
        langfuse_readonly,
    )


def generate_report_plan(
    *,
    topic: str,
    thesis: str,
    title: ReportTitleOutput,
    notes: List[EvidenceContributionNote],
    evidence_records: List[EvidenceRecord],
    warmstart_pairs: List[WarmstartPair],
    model: str,
    langfuse_readonly: bool = False,
) -> ReportPlanOutput:
    note_lines = []
    for note in notes:
        evidence = next((e for e in evidence_records if e.evidence_id == note.evidence_id), None)
        citation = evidence.citation_id if evidence else -1
        note_lines.append(
            f"- Evidence {note.evidence_id} [citation {citation}] ({note.stance}): "
            f"{note.contribution_summary} | angles={note.key_angles_for_report}"
        )
    note_digest = "\n".join(note_lines) if note_lines else "No evidence notes available."

    warmstart_digest = []
    for pair in warmstart_pairs:
        q = normalize_whitespace(pair.question)
        a = normalize_whitespace(pair.answer)
        if q or a:
            warmstart_digest.append(f"- Q: {q} | A: {a}")
    warmstart_text = "\n".join(warmstart_digest) if warmstart_digest else "No warmstart context."

    valid_ids = [e.evidence_id for e in evidence_records]

    plan = _langfuse_call(
        "staged_report_plan",
        {
            "topic": topic,
            "thesis": thesis,
            "title": title.title,
            "subtitle": title.subtitle,
            "editorial_angle": title.editorial_angle,
            "note_digest": note_digest,
            "warmstart_text": warmstart_text,
            "valid_ids": str(valid_ids),
        },
        ReportPlanOutput,
        model,
        langfuse_readonly,
    )

    valid_set = set(valid_ids)
    assert len(valid_ids) > 0
    for idx, section in enumerate(plan.sections, 1):
        if not section.section_id:
            section.section_id = f"S{idx}"
        cleaned = [eid for eid in section.must_include_evidence_ids if eid in valid_set]
        if not cleaned:
            cleaned = valid_ids
        section.must_include_evidence_ids = cleaned
        if not section.web_queries:
            section.web_queries = []

    if not plan.sections:
        raise ValueError("No sections generated for the report plan.")

    used_ids: Set[int] = set()
    for section in plan.sections:
        used_ids.update(section.must_include_evidence_ids)
    missing_ids = [eid for eid in valid_ids if eid not in used_ids]
    for missing_id in missing_ids:
        target = min(plan.sections, key=lambda s: len(s.must_include_evidence_ids))
        target.must_include_evidence_ids.append(missing_id)
    for section in plan.sections:
        section.must_include_evidence_ids = list(dict.fromkeys(section.must_include_evidence_ids))
    return plan


def draft_section(
    *,
    topic: str,
    thesis: str,
    title: ReportTitleOutput,
    section: SectionPlanItem,
    core_packet: str,
    core_citations: List[int],
    web_packet: str,
    web_citations: List[int],
    model: str,
    target_words: int = 350,
    langfuse_readonly: bool = False,
) -> SectionDraftOutput:
    allowed = sorted(set(core_citations + web_citations))
    out = _langfuse_call(
        "staged_report_section_draft",
        {
            "topic": topic,
            "thesis": thesis,
            "report_title": title.title,
            "section_id": section.section_id,
            "heading": section.heading,
            "purpose": section.purpose,
            "key_points": str(section.key_points),
            "storytelling_moves": str(section.storytelling_moves),
            "allowed": str(allowed),
            "core_packet": core_packet,
            "web_packet": web_packet,
            "target_words": target_words,
        },
        SectionDraftOutput,
        model,
        langfuse_readonly,
    )
    out.section_id = section.section_id
    out.heading = section.heading
    out.section_markdown = separate_inline_citations(out.section_markdown)

    allowed_set = set(allowed)
    used = find_citation_ids(out.section_markdown)
    if not used.issubset(allowed_set):
        out.section_markdown = remove_invalid_citations(out.section_markdown, allowed_set)
        used = find_citation_ids(out.section_markdown)
    out.used_citations = sorted(used)
    return out


class EntailmentCheck(BaseModel):
    is_entailed: bool
    issue: str  # empty string if entailed, description of problem if not


def fact_check_section(
    *,
    drafted: SectionDraftOutput,
    section: SectionPlanItem,
    core_packet: str,
    web_packet: str,
    evidence_records: List[EvidenceRecord],
    registry: SourceRegistry,
    topic: str,
    thesis: str,
    title: ReportTitleOutput,
    model: str,
    target_words: int = 350,
    langfuse_readonly: bool = False,
) -> Tuple[SectionDraftOutput, Dict[str, Any]]:
    """Fact-check a drafted section sentence-by-sentence and revise if issues are found."""
    if not drafted.used_citations:
        return drafted, {"total_checked": 0, "issues_found": 0}

    # Build citation_id → evidence text lookup for citations used in this section
    evidence_by_cid = {e.citation_id: e for e in evidence_records}
    citation_id_to_text: Dict[int, str] = {}
    for cid in drafted.used_citations:
        if cid in evidence_by_cid:
            e = evidence_by_cid[cid]
            citation_id_to_text[cid] = (
                f"[{cid}] ({e.source_type.upper()}) Question: {e.question}\nFinding: {e.finding}"
            )
        else:
            try:
                src = registry.get(cid)
                finding = src.description or (src.snippets[0] if src.snippets else "")
                if finding:
                    citation_id_to_text[cid] = f"[{cid}] (WEB) Title: {src.title}\nFinding: {finding}"
            except Exception:
                pass

    if not citation_id_to_text:
        return drafted, {"total_checked": 0, "issues_found": 0}

    # Parse section_markdown into (sentence, citation_ids) tasks
    parts = re.split(r'(\[\d+\])', drafted.section_markdown)
    tasks: List[Tuple[str, List[int]]] = []
    i = 0
    while i < len(parts) - 1:
        text_chunk = parts[i]
        citation_token = parts[i + 1]
        if not text_chunk.strip():
            i += 2
            continue
        m = re.match(r'\[(\d+)\]', citation_token)
        if not m:
            i += 2
            continue
        N = int(m.group(1))
        if N not in citation_id_to_text:
            i += 2
            continue
        sentence = text_chunk + citation_token
        indices = [N]
        j = i + 2
        while j + 1 < len(parts) and not parts[j].strip():
            cm = re.match(r'\[(\d+)\]', parts[j + 1])
            if not cm:
                break
            sentence += parts[j + 1]
            extra_N = int(cm.group(1))
            if extra_N in citation_id_to_text:
                indices.append(extra_N)
            j += 2
        tasks.append((sentence, indices))
        i += 2

    if not tasks:
        return drafted, {"total_checked": 0, "issues_found": 0}

    llm = get_llm(model_name=model, temperature=0)

    async def _check_all() -> List[Optional[Tuple[str, str]]]:
        async def check_one(sentence: str, indices: List[int]) -> Optional[Tuple[str, str]]:
            sources = "\n\n".join(citation_id_to_text[n] for n in indices)
            response = await call_llm_with_structured_output(
                "fact_check_sentence",
                {"sentence": sentence, "sources": sources},
                EntailmentCheck,
                llm,
                langfuse_readonly=langfuse_readonly,
            )
            if response and not response.is_entailed:
                return sentence, response.issue
            return None
        return await asyncio.gather(*[check_one(s, idxs) for s, idxs in tasks])

    results = asyncio.run(_check_all())
    criticisms = [r for r in results if r is not None]
    stats: Dict[str, Any] = {"total_checked": len(tasks), "issues_found": len(criticisms)}
    print(f"  [fact_check] section {section.section_id}: {len(criticisms)}/{len(tasks)} issue(s)")

    if not criticisms:
        return drafted, stats

    criticisms_str = json.dumps(
        [{"original_sentence": sentence, "criticism": issue} for sentence, issue in criticisms],
        indent=2,
    )
    allowed = sorted(set(drafted.used_citations))
    try:
        revised = _langfuse_call(
        "staged_report_section_revise",
        {
            "topic": topic,
            "thesis": thesis,
            "report_title": title.title,
            "section_id": section.section_id,
            "heading": section.heading,
            "purpose": section.purpose,
            "key_points": str(section.key_points),
            "storytelling_moves": str(section.storytelling_moves),
            "allowed": str(allowed),
            "core_packet": core_packet,
            "web_packet": web_packet,
            "previous_draft": drafted.section_markdown,
            "criticisms": criticisms_str,
            "target_words": target_words,
        },
        SectionDraftOutput,
        model,
        langfuse_readonly,
    )
    except RuntimeError as e:
        print(f"  [fact_check] section {section.section_id}: revision failed ({e}); keeping original.")
        return drafted, stats

    revised.section_id = section.section_id
    revised.heading = section.heading
    revised.section_markdown = separate_inline_citations(revised.section_markdown)
    allowed_set = set(allowed)
    used = find_citation_ids(revised.section_markdown)
    if not used.issubset(allowed_set):
        revised.section_markdown = remove_invalid_citations(revised.section_markdown, allowed_set)
        used = find_citation_ids(revised.section_markdown)
    revised.used_citations = sorted(used)
    return revised, stats


def final_polish_report(
    *,
    topic: str,
    thesis: str,
    title: ReportTitleOutput,
    plan: ReportPlanOutput,
    draft_markdown: str,
    allowed_citations: Set[int],
    model: str,
    target_total_words: int = 1500,
    langfuse_readonly: bool = False,
) -> str:
    out = _langfuse_call(
        "staged_report_final_polish",
        {
            "topic": topic,
            "thesis": thesis,
            "title": title.title,
            "subtitle": title.subtitle,
            "plan_json": plan.model_dump_json(indent=2),
            "allowed_citations": str(sorted(allowed_citations)),
            "draft_markdown": draft_markdown,
            "target_total_words": target_total_words,
        },
        FinalReportOutput,
        model,
        langfuse_readonly,
    )
    polished = out.report_markdown.strip()
    polished = strip_llm_reference_sections(polished)
    polished = remove_invalid_citations(polished, allowed_citations)
    return polished


# ---------------------------------------------------------------------------
# Top-level pipeline entry point
# ---------------------------------------------------------------------------

def generate_report_from_data(
    *,
    topic: str,
    thesis: str,
    tree: Dict[str, Any],
    warmstart_data: List[Dict[str, Any]],
    run_dir: str,
    db_description: str = "",
    model: str = "gpt-5",
    output_file: str = "co_storm_report_staged.md",
    serper_query_params: Optional[Dict[str, Any]] = None,
    disable_web_enrichment: bool = True,
    langfuse_readonly: bool = False,
    evidence_records: Optional[List[EvidenceRecord]] = None,
    registry: Optional[SourceRegistry] = None,
    target_words_per_section: int = 350,
    target_total_words: int = 1500,
    allow_empty_thesis: bool = False,
) -> Dict[str, str]:
    """Core pipeline that operates on in-memory data. Called directly from the live pipeline
    or via generate_report() when loading from disk.

    allow_empty_thesis: when True (e.g. the no_thesis ablation, which runs with
    --skip_thesis), an empty thesis is permitted and the report is generated
    thesis-free (topic-only framing). When False, an empty thesis is treated as a
    failure of thesis generation and raises, so genuine failures still surface.
    """
    run_dir_path = Path(run_dir).resolve()

    if not topic:
        raise RuntimeError("'topic' is required")
    if not thesis and not allow_empty_thesis:
        raise RuntimeError("'thesis' is empty — has thesis generation completed?")

    if registry is None:
        registry = SourceRegistry()
    if evidence_records is None:
        evidence_records = collect_tree_evidence(tree, registry)
    if not evidence_records:
        raise RuntimeError("No selected evidence was found in tree (is_final_selected=true).")

    warmstart_pairs = parse_warmstart_pairs(warmstart_data)

    print(f"[stage] topic: {topic}")
    print(f"[stage] thesis: {thesis}")
    print(f"[stage] core evidence items: {len(evidence_records)}")

    evidence_notes: List[EvidenceContributionNote] = []
    for evidence in evidence_records:
        note = generate_evidence_note(
            topic=topic,
            thesis=thesis,
            evidence=evidence,
            model=model,
            langfuse_readonly=langfuse_readonly,
        )
        evidence_notes.append(note)
        print(f"[stage] evidence note generated for evidence_id={evidence.evidence_id}")

    title = generate_title(
        topic=topic,
        thesis=thesis,
        notes=evidence_notes,
        model=model,
        langfuse_readonly=langfuse_readonly,
    )
    print(f"[stage] title generated: {title.title}")

    plan = generate_report_plan(
        topic=topic,
        thesis=thesis,
        title=title,
        notes=evidence_notes,
        evidence_records=evidence_records,
        warmstart_pairs=warmstart_pairs,
        model=model,
        langfuse_readonly=langfuse_readonly,
    )
    print(f"[stage] plan generated with {len(plan.sections)} sections")

    serper_rm = init_serper_rm(serper_query_params) if not disable_web_enrichment else None
    if serper_rm is None and not disable_web_enrichment:
        print("[web] SERPER_API_KEY missing or retriever unavailable; continuing without web enrichment.")

    note_map = {n.evidence_id: n for n in evidence_notes}
    drafted_sections: List[SectionDraftOutput] = []
    fact_check_accumulator: List[Dict[str, Any]] = []

    for section in plan.sections:
        core_packet, core_citations = build_core_evidence_packet(
            section=section,
            evidence_records=evidence_records,
            note_map=note_map,
        )
        web_citations = retrieve_web_context_for_section(
            section=section,
            serper_rm=serper_rm,
            registry=registry,
        )
        web_packet = build_web_packet(web_citations, registry)

        drafted = draft_section(
            topic=topic,
            thesis=thesis,
            title=title,
            section=section,
            core_packet=core_packet,
            core_citations=core_citations,
            web_packet=web_packet,
            web_citations=web_citations,
            model=model,
            target_words=target_words_per_section,
            langfuse_readonly=langfuse_readonly,
        )
        drafted, section_stats = fact_check_section(
            drafted=drafted,
            section=section,
            core_packet=core_packet,
            web_packet=web_packet,
            evidence_records=evidence_records,
            registry=registry,
            topic=topic,
            thesis=thesis,
            title=title,
            model=model,
            target_words=target_words_per_section,
            langfuse_readonly=langfuse_readonly,
        )
        fact_check_accumulator.append(section_stats)
        drafted_sections.append(drafted)
        print(
            f"[stage] drafted section {section.section_id} with "
            f"{len(drafted.used_citations)} citations"
        )

    fact_check_stats: Dict[str, Any] = {
        "total_checked": sum(s["total_checked"] for s in fact_check_accumulator),
        "issues_found": sum(s["issues_found"] for s in fact_check_accumulator),
        "per_section": [
            {"section_id": plan.sections[i].section_id, **fact_check_accumulator[i]}
            for i in range(len(plan.sections))
        ],
    }

    pre_final_draft = build_pre_final_draft(
        thesis=thesis,
        title=title,
        plan=plan,
        sections=drafted_sections,
    )

    available_citations = registry.available_ids()
    final_report_body = final_polish_report(
        topic=topic,
        thesis=thesis,
        title=title,
        plan=plan,
        draft_markdown=pre_final_draft,
        allowed_citations=available_citations,
        model=model,
        target_total_words=target_total_words,
        langfuse_readonly=langfuse_readonly,
    )

    final_report_body = separate_inline_citations(final_report_body)
    final_report_body = remove_invalid_citations(final_report_body, available_citations)
    final_report_body, old_to_new_map = reindex_citations_by_appearance(final_report_body)
    new_to_old_map = {new: old for old, new in old_to_new_map.items()}

    used_citations = find_citation_ids(final_report_body)
    appendix = build_sources_appendix(
        used_citations,
        registry,
        citation_source_map=new_to_old_map,
    )
    final_report = final_report_body + ("\n" + appendix if appendix else "")

    output_path = run_dir_path / output_file
    output_path.write_text(final_report, encoding="utf-8")

    notes_path = run_dir_path / "staged_report_notes.json"
    title_path = run_dir_path / "staged_report_title.json"
    plan_path = run_dir_path / "staged_report_plan.json"
    sections_path = run_dir_path / "staged_report_sections.json"
    provenance_path = run_dir_path / "staged_report_provenance.json"

    note_payload = []
    evidence_by_id = {e.evidence_id: e for e in evidence_records}
    for note in evidence_notes:
        e = evidence_by_id.get(note.evidence_id)
        note_payload.append(
            {
                "evidence": asdict(e) if e else {},
                "note": note.model_dump(),
            }
        )
    fact_check_path = run_dir_path / "fact_check_summary.json"
    write_json(notes_path, note_payload)
    write_json(title_path, title.model_dump())
    write_json(plan_path, plan.model_dump())
    write_json(sections_path, [s.model_dump() for s in drafted_sections])
    write_json(fact_check_path, fact_check_stats)
    write_json(
        provenance_path,
        {
            "created_at_utc": datetime.datetime.utcnow().isoformat(),
            "run_dir": str(run_dir_path),
            "topic": topic,
            "thesis": thesis,
            "db_description": db_description,
            "model": model,
            "core_evidence_count": len(evidence_records),
            "warmstart_pairs_count": len(warmstart_pairs),
            "used_citation_ids": sorted(used_citations),
            "citation_reindex_map_old_to_new": {
                str(old): new
                for old, new in sorted(old_to_new_map.items(), key=lambda item: item[1])
            },
            "sources": [asdict(src) for src in registry.all_records()],
            "artifacts": {
                "report": str(output_path),
                "notes": str(notes_path),
                "title": str(title_path),
                "plan": str(plan_path),
                "sections": str(sections_path),
                "provenance": str(provenance_path),
                "fact_check_summary": str(fact_check_path),
            },
        },
    )

    print(f"[done] report: {output_path}")
    return {
        "report": str(output_path),
        "report_content": final_report,
        "citation_old_to_new_map": old_to_new_map,
        "fact_check_stats": fact_check_stats,
        "notes": str(notes_path),
        "title": str(title_path),
        "plan": str(plan_path),
        "sections": str(sections_path),
        "provenance": str(provenance_path),
        "fact_check_summary": str(fact_check_path),
    }


def generate_report(
    run_dir: str,
    *,
    model: str = "gpt-5",
    output_file: str = "co_storm_report_staged.md",
    serper_query_params: Optional[str] = None,
    disable_web_enrichment: bool = True,
    langfuse_readonly: bool = False,
    target_words_per_section: int = 350,
    target_total_words: int = 1500,
) -> Dict[str, str]:
    """Load tree/warmstart/metadata from disk and run the staged pipeline."""
    run_dir_path = Path(run_dir).resolve()
    tree_path = run_dir_path / "tree.json"
    warmstart_path = run_dir_path / "warmstart_conversation.json"
    metadata_path = run_dir_path / "run_metadata.json"

    if not tree_path.exists():
        raise FileNotFoundError(f"tree.json not found: {tree_path}")
    if not warmstart_path.exists():
        raise FileNotFoundError(f"warmstart_conversation.json not found: {warmstart_path}")

    tree = read_json(tree_path)
    warmstart_data = read_json(warmstart_path)
    metadata = read_json(metadata_path) if metadata_path.exists() else {}

    topic = metadata.get("topic", "")
    thesis = tree.get("thesis", "")
    db_description = metadata.get("db_description") or DB_DESCRIPTION_MAP.get(metadata.get("domain", ""), "")

    serper_params: Optional[Dict[str, Any]] = None
    if serper_query_params:
        try:
            serper_params = json.loads(serper_query_params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid serper_query_params JSON: {e}") from e
    elif isinstance(metadata.get("serper_query_params"), dict):
        serper_params = metadata["serper_query_params"]

    # Pre-build registry + evidence list combining warmstart internet sources (if serialized)
    # and tree-selected evidence, matching the assembly order of the live pipeline.
    registry = SourceRegistry()
    warmstart_records = load_warmstart_evidence_from_disk(warmstart_data, registry, evidence_id_start=1)
    if warmstart_records:
        print(f"[generate_report] Loaded {len(warmstart_records)} warmstart evidence records from disk")
    else:
        print("[generate_report] No warmstart cited_info on disk (old run or warmstart had no web sources); using tree evidence only")
    tree_records = collect_tree_evidence(tree, registry)
    # Re-index tree evidence_ids to follow warmstart records
    for i, rec in enumerate(tree_records):
        rec.evidence_id = len(warmstart_records) + i + 1
    evidence_records = warmstart_records + tree_records

    return generate_report_from_data(
        topic=topic,
        thesis=thesis,
        tree=tree,
        warmstart_data=warmstart_data,
        run_dir=run_dir,
        db_description=db_description,
        model=model,
        output_file=output_file,
        serper_query_params=serper_params,
        evidence_records=evidence_records,
        registry=registry,
        disable_web_enrichment=disable_web_enrichment,
        langfuse_readonly=langfuse_readonly,
        target_words_per_section=target_words_per_section,
        target_total_words=target_total_words,
    )
