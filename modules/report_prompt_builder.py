"""
modules/report_prompt_builder.py — Presales Report Prompt Engineering
───────────────────────────────────────────────────────────────────────
Responsibilities:
  - Provide structured prompt templates for each report type:
      • Executive Summary
      • Competitive Analysis
      • Requirements Extraction
      • Risk Assessment
      • Proposal Outline
      • Custom / Free-form Analysis
  - Format document context for report-oriented LLM prompts
  - Return section-aware prompts that produce structured markdown output

Design:
  Each report type has:
    1. A SYSTEM INSTRUCTION block defining the analyst persona
    2. A TASK block describing exactly what to produce
    3. A FORMAT block specifying expected output structure
    4. The CONTEXT block (injected at build time)
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Report type registry
# ──────────────────────────────────────────────

REPORT_TYPES = {
    "executive_summary": "Executive Summary",
    "competitive_analysis": "Competitive Analysis",
    "requirements_extraction": "Requirements Extraction",
    "risk_assessment": "Risk & Gap Assessment",
    "proposal_outline": "Proposal Outline",
    "custom": "Custom Analysis",
}


@dataclass
class ReportPromptConfig:
    report_type: str
    documents: List[str]          # source document names
    custom_instruction: str = ""  # extra context or focus area from user
    audience: str = "presales team"
    max_context_chars: int = 8000


class ReportPromptBuilder:
    """
    Builds structured LLM prompts for presales report generation.

    Usage:
        builder = ReportPromptBuilder()
        prompt = builder.build(context_text, config)
    """

    # ──────────────────────────────────────────
    # Shared analyst persona
    # ──────────────────────────────────────────

    _ANALYST_PERSONA = (
        "You are a senior presales solutions analyst with 15+ years of experience "
        "producing structured, insight-rich reports for enterprise sales engagements. "
        "You write clearly, concisely, and professionally. "
        "You extract actionable intelligence from raw documents and present it "
        "in a format ready for executive or client-facing delivery."
    )

    _GROUNDING_RULE = (
        "STRICT RULE: Base your analysis ONLY on the provided document context. "
        "Do not fabricate facts, invent figures, or assume information not present. "
        "If a section cannot be completed due to missing information, "
        "explicitly note: '[Insufficient data in documents]'."
    )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def build(
        self,
        context_text: str,
        config: ReportPromptConfig,
    ) -> str:
        """
        Build the full report generation prompt.

        Args:
            context_text: Formatted document context (pre-chunked/retrieved).
            config: Report configuration object.

        Returns:
            Formatted prompt string ready for the LLM.
        """
        template_fn = {
            "executive_summary":      self._executive_summary_prompt,
            "competitive_analysis":   self._competitive_analysis_prompt,
            "requirements_extraction":self._requirements_extraction_prompt,
            "risk_assessment":        self._risk_assessment_prompt,
            "proposal_outline":       self._proposal_outline_prompt,
            "custom":                 self._custom_analysis_prompt,
        }.get(config.report_type, self._custom_analysis_prompt)

        task_block = template_fn(config)
        prompt = self._assemble(context_text, task_block, config)

        logger.debug(
            f"Built report prompt: type={config.report_type}, "
            f"len={len(prompt)} chars"
        )
        return prompt

    def build_context_from_texts(
        self,
        texts: List[str],
        sources: List[str],
        max_chars: int = 8000,
    ) -> str:
        """
        Format raw text chunks into a context block for the report prompt.

        Args:
            texts: List of document text chunks.
            sources: Corresponding source file names.
            max_chars: Maximum total context length.

        Returns:
            Formatted <context> block string.
        """
        sections = []
        total = 0

        for i, (text, src) in enumerate(zip(texts, sources), 1):
            header = f"[Document {i}] Source: {src}"
            section = f"{header}\n{text.strip()}"
            if total + len(section) > max_chars:
                remaining = max_chars - total - len(header) - 20
                if remaining > 100:
                    section = f"{header}\n{text[:remaining]}... [truncated]"
                    sections.append(section)
                break
            sections.append(section)
            total += len(section)

        joined = "\n\n---\n\n".join(sections)
        return f"<context>\n{joined}\n</context>"

    # ──────────────────────────────────────────
    # Report type templates
    # ──────────────────────────────────────────

    def _executive_summary_prompt(self, config: ReportPromptConfig) -> str:
        return f"""
TASK: Generate a concise Executive Summary suitable for {config.audience}.

The summary must cover:

## Executive Summary

### 1. Overview
- What this document/set of documents is about
- Key subject, product, or initiative described

### 2. Key Highlights
- Top 3–5 most important facts or findings
- Business value or strategic relevance

### 3. Critical Numbers & Data Points
- Any significant metrics, KPIs, timelines, or financial figures mentioned

### 4. Recommendations
- What action the presales team should take based on this material

### 5. Open Questions
- Gaps or areas needing clarification before a proposal can proceed

{f'FOCUS AREA: {config.custom_instruction}' if config.custom_instruction else ''}

FORMAT: Respond in clean markdown with the exact section headers above.
LENGTH: Aim for 400–700 words. Be precise — avoid filler sentences.
"""

    def _competitive_analysis_prompt(self, config: ReportPromptConfig) -> str:
        return f"""
TASK: Produce a Competitive Analysis report for the {config.audience}.

Structure your report as follows:

## Competitive Analysis Report

### 1. Market Context
- Industry or domain mentioned in the documents
- Market dynamics or trends referenced

### 2. Competitors Identified
For each competitor or alternative solution mentioned:
- **Name / Product**
- Strengths cited
- Weaknesses or gaps cited
- Positioning vs. our solution (if inferable)

### 3. Differentiators
- What unique advantages are described in these documents
- Key selling points for the presales narrative

### 4. Threats & Opportunities
- Competitive threats identified
- Market opportunities to exploit

### 5. Win Themes
- 2–3 concise messages the presales team should lead with

{f'FOCUS AREA: {config.custom_instruction}' if config.custom_instruction else ''}

FORMAT: Respond in clean markdown. Use tables where comparisons are side-by-side.
LENGTH: Be thorough but avoid padding. Quality over quantity.
"""

    def _requirements_extraction_prompt(self, config: ReportPromptConfig) -> str:
        return f"""
TASK: Extract and structure all requirements mentioned across the documents for the {config.audience}.

## Requirements Extraction Report

### 1. Functional Requirements
List every functional requirement found. For each:
- **ID**: REQ-F-XXX
- **Description**: What the system/solution must do
- **Source**: Which document / section it came from
- **Priority**: High / Medium / Low (infer from language used, e.g. "must", "should", "nice to have")

### 2. Non-Functional Requirements
Same format as above for NFRs (performance, security, scalability, compliance, etc.)
- **ID**: REQ-NF-XXX

### 3. Integration Requirements
Any APIs, systems, or third-party tools mentioned that need to integrate.

### 4. Constraints
Budget, timeline, regulatory, or technical constraints explicitly mentioned.

### 5. Assumptions Made
List any assumptions you had to make due to ambiguous or missing information.

{f'FOCUS AREA: {config.custom_instruction}' if config.custom_instruction else ''}

FORMAT: Respond in structured markdown. Use numbered lists for requirements.
"""

    def _risk_assessment_prompt(self, config: ReportPromptConfig) -> str:
        return f"""
TASK: Produce a Risk & Gap Assessment for the {config.audience} based on the documents.

## Risk & Gap Assessment

### 1. Identified Risks
For each risk found or inferred:
| Risk | Category | Likelihood | Impact | Mitigation |
|------|----------|------------|--------|------------|
(fill in the table; categories: Technical, Commercial, Operational, Regulatory, Timeline)

### 2. Capability Gaps
- What is the client/prospect asking for that may not be immediately available
- Gaps between stated requirements and what is described in the documents

### 3. Dependencies & Blockers
- External dependencies mentioned (third parties, data sources, approvals)
- Blockers that could delay a deal or delivery

### 4. Compliance & Regulatory Flags
- Any regulations, standards, or certifications mentioned (GDPR, ISO, SOC2, etc.)
- Red flags the presales team must escalate

### 5. Presales Action Items
- Specific risks the team must address before submitting a proposal

{f'FOCUS AREA: {config.custom_instruction}' if config.custom_instruction else ''}

FORMAT: Respond in structured markdown with the table and section headers above.
"""

    def _proposal_outline_prompt(self, config: ReportPromptConfig) -> str:
        return f"""
TASK: Generate a Proposal Outline the {config.audience} can use as a starting framework.

## Proposal Outline

### 1. Executive Summary (Draft)
- 2–3 sentence value proposition derived from the documents

### 2. Understanding of Requirements
- Restate the client's goals and needs as understood from the documents
- Show you understand their pain points

### 3. Proposed Solution Overview
- High-level solution approach (inferred from documents)
- Key capabilities to highlight

### 4. Differentiators & Value Drivers
- Why this solution vs. alternatives
- ROI or business value points to include

### 5. Delivery Approach
- Phases, milestones, or timelines mentioned
- Resource and skills considerations

### 6. Pricing Signals
- Any budget, cost, or pricing signals found in the documents
- Recommended pricing strategy (enterprise, per-seat, etc.)

### 7. Next Steps
- Recommended next steps for the presales team

### 8. Appendix Placeholders
- What supporting material should be attached (case studies, technical specs, etc.)

{f'FOCUS AREA: {config.custom_instruction}' if config.custom_instruction else ''}

FORMAT: Respond in structured markdown with clear headers.
NOTE: This is an outline/framework, not the final proposal copy.
"""

    def _custom_analysis_prompt(self, config: ReportPromptConfig) -> str:
        instruction = config.custom_instruction or (
            "Provide a comprehensive structured analysis of the documents. "
            "Identify key themes, data points, insights, and actionable recommendations."
        )
        return f"""
TASK: {instruction}

Audience: {config.audience}

Structure your response with:
- Clear section headers
- Bullet points for lists of items
- Tables where comparisons are useful
- A final "Key Takeaways" section with 3–5 actionable points

FORMAT: Clean professional markdown.
"""

    # ──────────────────────────────────────────
    # Assembly
    # ──────────────────────────────────────────

    def _assemble(
        self,
        context_text: str,
        task_block: str,
        config: ReportPromptConfig,
    ) -> str:
        doc_list = ", ".join(config.documents) if config.documents else "uploaded documents"
        return (
            f"{self._ANALYST_PERSONA}\n\n"
            f"{self._GROUNDING_RULE}\n\n"
            f"SOURCE DOCUMENTS: {doc_list}\n\n"
            f"DOCUMENT CONTEXT:\n{context_text}\n\n"
            f"{task_block.strip()}\n\n"
            f"BEGIN REPORT:\n"
        )
