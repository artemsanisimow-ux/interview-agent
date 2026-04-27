"""
User Interview Agent
====================
Processes interview transcripts using Teresa Torres' Continuous Discovery framework.

Pipeline (no LangGraph — linear, no cycles):
  load → extract (1 LLM call) → map to OST → generate guide (1 LLM call) → save

What it does:
1. TranscriptLoader  — accepts text, file path, or URL
2. QuoteExtractor    — extracts quotes + maps to opportunities (1 LLM call)
3. OST Mapper        — enriches CR agent's Opportunity Solution Tree
4. GuideGenerator    — generates next interview guide based on OST gaps (1 LLM call)
5. Persistence       — saves all to local DB + CR agent DB as signals

Setup:
    pip install langchain-anthropic python-dotenv requests

Usage:
    python3 interview_agent.py --lang en
    python3 interview_agent.py --lang ru
    python3 interview_agent.py --lang en --file transcript.txt
    python3 interview_agent.py --lang en --url https://notion.so/...
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import requests
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
from i18n import get_language, get_language_instruction, set_language
from i18n import t as tr

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MODEL    = ChatAnthropic(model="claude-opus-4-5", max_tokens=3000)
DB_PATH  = "interviews.db"
CR_DB    = os.getenv("CR_DB_PATH", "../cr-agent/cr_signals.db")

MIN_QUOTE_LENGTH  = 15   # chars — ignore very short quotes
MAX_TRANSCRIPT    = 20000  # chars sent to LLM
GUIDE_MAX_Q       = 8    # questions in generated guide


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class Quote:
    text:        str
    speaker:     str        # "user" | "interviewer" | "unknown"
    opportunity: str        # mapped OST opportunity
    theme:       str        # e.g. "onboarding", "export", "pricing"
    sentiment:   float      # -1.0 to 1.0
    insight:     str        # one-sentence interpretation
    assumption:  str        # assumption this challenges or confirms
    timestamp:   str = ""   # optional — from transcript markers like [00:03:22]

    @property
    def is_user_quote(self) -> bool:
        return self.speaker.lower() in ("user", "participant", "interviewee", "respondent")


@dataclass
class InterviewSession:
    session_id:   str
    participant:  str
    product:      str
    date:         str
    raw_text:     str
    lang:         str
    quotes:       list[Quote]       = field(default_factory=list)
    opportunities: list[str]        = field(default_factory=list)
    key_insights: list[str]         = field(default_factory=list)
    assumptions:  list[str]         = field(default_factory=list)
    created_at:   str               = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class InterviewGuide:
    session_id:   str
    based_on:     list[str]   # session IDs this guide is based on
    ost_gaps:     list[str]   # opportunities with weak evidence
    questions:    list[dict]  # {question, rationale, type}
    warm_up:      list[str]
    hypothesis:   str         # core assumption to test
    target_profile: str
    created_at:   str = field(default_factory=lambda: datetime.now().isoformat())


# ─────────────────────────────────────────────
# TRANSCRIPT SOURCE PROTOCOL
# ─────────────────────────────────────────────

@runtime_checkable
class TranscriptSource(Protocol):
    name: str
    def load(self) -> str: ...


class TextSource:
    name = "text"
    def __init__(self, text: str):
        self._text = text
    def load(self) -> str:
        return self._text.strip()


class FileSource:
    name = "file"
    def __init__(self, path: str):
        self._path = Path(path)
    def load(self) -> str:
        if not self._path.exists():
            raise FileNotFoundError(f"Transcript file not found: {self._path}")
        return self._path.read_text(encoding="utf-8").strip()


class URLSource:
    """Fetches transcript from a URL — Notion export, Google Doc export, etc."""
    name = "url"
    def __init__(self, url: str):
        self._url = url
    def load(self) -> str:
        try:
            r = requests.get(self._url, timeout=15,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            # Strip HTML tags if needed
            text = re.sub(r"<[^>]+>", " ", r.text)
            return re.sub(r"\s+", " ", text).strip()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch transcript from {self._url}: {e}")


# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL UNIQUE,
            participant  TEXT NOT NULL,
            product      TEXT NOT NULL,
            date         TEXT NOT NULL,
            lang         TEXT NOT NULL DEFAULT 'en',
            raw_text     TEXT,
            key_insights TEXT,
            assumptions  TEXT,
            created_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS quotes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            text         TEXT NOT NULL,
            speaker      TEXT NOT NULL,
            opportunity  TEXT NOT NULL,
            theme        TEXT NOT NULL,
            sentiment    REAL NOT NULL,
            insight      TEXT,
            assumption   TEXT,
            timestamp    TEXT,
            created_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS guides (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            based_on     TEXT NOT NULL,
            ost_gaps     TEXT NOT NULL,
            questions    TEXT NOT NULL,
            warm_up      TEXT,
            hypothesis   TEXT,
            target_profile TEXT,
            created_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_quotes_session ON quotes(session_id);
        CREATE INDEX IF NOT EXISTS idx_quotes_opp ON quotes(opportunity);
    """)
    conn.commit()
    conn.close()


def save_session(session: InterviewSession) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO sessions "
        "(session_id, participant, product, date, lang, raw_text, key_insights, assumptions) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (session.session_id, session.participant, session.product,
         session.date, session.lang,
         session.raw_text[:5000],  # store trimmed version
         json.dumps(session.key_insights, ensure_ascii=False),
         json.dumps(session.assumptions, ensure_ascii=False))
    )
    if session.quotes:
        conn.executemany(
            "INSERT INTO quotes "
            "(session_id, text, speaker, opportunity, theme, sentiment, insight, assumption, timestamp) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(session.session_id, q.text, q.speaker, q.opportunity,
              q.theme, q.sentiment, q.insight, q.assumption, q.timestamp)
             for q in session.quotes]
        )
    conn.commit()
    conn.close()


def save_guide(guide: InterviewGuide) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO guides "
        "(session_id, based_on, ost_gaps, questions, warm_up, hypothesis, target_profile) "
        "VALUES (?,?,?,?,?,?,?)",
        (guide.session_id,
         json.dumps(guide.based_on),
         json.dumps(guide.ost_gaps, ensure_ascii=False),
         json.dumps(guide.questions, ensure_ascii=False),
         json.dumps(guide.warm_up, ensure_ascii=False),
         guide.hypothesis, guide.target_profile)
    )
    conn.commit()
    conn.close()


def get_past_sessions(limit: int = 10) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT session_id, participant, product, date, key_insights "
        "FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_ost_opportunities(limit: int = 20) -> list[dict]:
    """Get opportunities ranked by quote count across all sessions."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT opportunity, theme, COUNT(*) as quote_count, "
        "AVG(sentiment) as avg_sentiment "
        "FROM quotes GROUP BY opportunity "
        "ORDER BY quote_count DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    s = {
        "sessions": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
        "quotes":   conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0],
        "guides":   conn.execute("SELECT COUNT(*) FROM guides").fetchone()[0],
        "opportunities": conn.execute(
            "SELECT COUNT(DISTINCT opportunity) FROM quotes"
        ).fetchone()[0],
    }
    conn.close()
    return s


def push_to_cr(session: InterviewSession) -> int:
    """Write user quotes as signals to CR agent's DB."""
    user_quotes = [q for q in session.quotes if q.is_user_quote]
    if not user_quotes:
        return 0

    cr = Path(CR_DB)
    if not cr.exists():
        cr = Path("cr_signals.db")

    saved = 0
    try:
        conn = sqlite3.connect(str(cr))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT,
                raw_text TEXT, opportunity TEXT, sentiment REAL,
                tags_json TEXT, created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT UNIQUE,
                outcome TEXT DEFAULT '', signal_count INTEGER DEFAULT 1,
                avg_sentiment REAL, status TEXT DEFAULT 'open',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
        """)
        for q in user_quotes:
            conn.execute(
                "INSERT INTO signals (source, raw_text, opportunity, sentiment, tags_json) "
                "VALUES (?,?,?,?,?)",
                ("user_interview", q.text[:500], q.opportunity, q.sentiment,
                 json.dumps(["interview", q.theme, session.participant[:30]]))
            )
            conn.execute(
                "INSERT INTO opportunities (title, signal_count, avg_sentiment) "
                "VALUES (?,1,?) ON CONFLICT(title) DO UPDATE SET "
                "signal_count=signal_count+1, updated_at=datetime('now')",
                (q.opportunity, q.sentiment)
            )
            saved += 1
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"   ⚠️  CR push: {e}")
    return saved


# ─────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────

def _call(system: str, user: str) -> dict:
    resp = MODEL.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"error": "parse_failed", "raw": raw[:200]}


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

EXTRACT_PROMPT = """{lang}
You are a product researcher applying Teresa Torres' Continuous Discovery Habits.
Extract structured insights from this user interview transcript.

Rules:
- Only extract direct quotes or close paraphrases from the USER (not the interviewer)
- Map each quote to a specific opportunity (user need or pain point, 5-8 words)
- Sentiment: -1.0 (very negative) to 1.0 (very positive)
- Insight: what this reveals about the user's underlying need
- Assumption: what product assumption this confirms or challenges
- Theme: single word category (onboarding, export, pricing, search, etc.)
- Timestamp: extract if visible in transcript (e.g. [00:03:22]), else empty string

Return ONLY valid JSON:
{{
  "quotes": [{{
    "text": str,
    "speaker": "user",
    "opportunity": str,
    "theme": str,
    "sentiment": float,
    "insight": str,
    "assumption": str,
    "timestamp": str
  }}],
  "key_insights": [str],
  "assumptions_challenged": [str],
  "top_opportunities": [str],
  "participant_profile": str
}}"""

GUIDE_PROMPT = """{lang}
You are a product researcher designing the next user interview guide.
Based on past interview findings and OST gaps, generate a focused guide.

Teresa Torres principles:
- One clear outcome to research
- Questions reveal opportunities, not validate solutions
- "Walk me through" and "Tell me about a time" over yes/no questions
- Follow emotion — probe when participant shows frustration or delight

OST opportunities with weak evidence (explore these):
{ost_gaps}

Key insights from past interviews:
{insights}

Assumptions still unchallenged:
{assumptions}

Return ONLY valid JSON:
{{
  "target_profile": str,
  "hypothesis": str,
  "warm_up": [str],
  "questions": [{{
    "question": str,
    "rationale": str,
    "type": "story|exploration|reaction|clarification"
  }}],
  "probes": [str],
  "things_to_avoid": [str]
}}"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def L(en: str, ru: str) -> str:
    return en if get_language() == "en" else ru


def _clean_transcript(text: str) -> str:
    """Normalize whitespace, keep timestamps."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate(text: str, max_chars: int = MAX_TRANSCRIPT) -> str:
    if len(text) <= max_chars:
        return text
    # Keep beginning and end — most valuable parts
    half = max_chars // 2
    return text[:half] + "\n\n[... transcript truncated ...]\n\n" + text[-half:]


# ─────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────

def load_transcript(source: TranscriptSource) -> str:
    text = source.load()
    if len(text) < MIN_QUOTE_LENGTH:
        raise ValueError(f"Transcript too short ({len(text)} chars)")
    return _clean_transcript(text)


def extract_quotes(transcript: str, participant: str,
                   product: str, lang: str) -> dict:
    """LLM call 1: extract quotes + map to opportunities."""
    system = EXTRACT_PROMPT.format(lang=get_language_instruction())
    user   = (f"Participant: {participant}\nProduct: {product}\n\n"
              f"TRANSCRIPT:\n{_truncate(transcript)}")
    return _call(system, user)


def build_session(session_id: str, participant: str, product: str,
                  lang: str, transcript: str, extracted: dict) -> InterviewSession:
    quotes = [
        Quote(
            text=q.get("text", ""),
            speaker=q.get("speaker", "user"),
            opportunity=q.get("opportunity", "Unclassified"),
            theme=q.get("theme", "other"),
            sentiment=float(q.get("sentiment", 0.0)),
            insight=q.get("insight", ""),
            assumption=q.get("assumption", ""),
            timestamp=q.get("timestamp", ""),
        )
        for q in extracted.get("quotes", [])
        if len(q.get("text", "")) >= MIN_QUOTE_LENGTH
    ]
    return InterviewSession(
        session_id=session_id,
        participant=participant,
        product=product,
        date=datetime.now().strftime("%Y-%m-%d"),
        raw_text=transcript,
        lang=lang,
        quotes=quotes,
        opportunities=extracted.get("top_opportunities", []),
        key_insights=extracted.get("key_insights", []),
        assumptions=extracted.get("assumptions_challenged", []),
    )


def generate_guide(session_id: str, based_on: list[str]) -> InterviewGuide:
    """LLM call 2: generate next interview guide from OST gaps."""
    ost = get_ost_opportunities(20)
    past = get_past_sessions(5)

    # Identify gaps — opportunities with few quotes or low evidence
    gaps = [o["opportunity"] for o in ost
            if o["quote_count"] < 3][:6]

    # Fallback if no OST data yet
    if not gaps and ost:
        gaps = [o["opportunity"] for o in ost[:4]]
    elif not gaps:
        gaps = ["General product experience", "Onboarding friction", "Core workflow pain points"]

    insights = []
    assumptions = []
    for s in past:
        if s.get("key_insights"):
            try:
                insights.extend(json.loads(s["key_insights"])[:2])
            except Exception:
                pass

    system = GUIDE_PROMPT.format(
        lang=get_language_instruction(),
        ost_gaps="\n".join(f"- {g}" for g in gaps),
        insights="\n".join(f"- {i}" for i in insights[:8]) or "None yet",
        assumptions="\n".join(f"- {a}" for a in assumptions[:5]) or "None yet",
    )
    user = L(
        "Generate the next interview guide focusing on the weakest evidence areas.",
        "Сгенерируй guide для следующего интервью, фокусируясь на областях с наименьшими доказательствами.",
    )
    result = _call(system, user)

    return InterviewGuide(
        session_id=session_id,
        based_on=based_on,
        ost_gaps=gaps,
        questions=result.get("questions", [])[:GUIDE_MAX_Q],
        warm_up=result.get("warm_up", []),
        hypothesis=result.get("hypothesis", ""),
        target_profile=result.get("target_profile", ""),
    )


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────

def render_session_report(session: InterviewSession,
                           guide: InterviewGuide) -> str:
    lang = get_language()
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")

    if lang == "en":
        lines = [
            f"# Interview Report: {session.participant}",
            f"Product: {session.product} | Date: {session.date} | Session: {session.session_id}",
            "",
            "## Key Insights",
        ]
        for i in session.key_insights:
            lines.append(f"- {i}")

        lines += ["", "## Assumptions Challenged"]
        for a in session.assumptions:
            lines.append(f"- {a}")

        user_quotes = [q for q in session.quotes if q.is_user_quote]
        lines += ["", f"## Quotes ({len(user_quotes)} from user)"]
        for q in user_quotes[:15]:
            lines.append(
                f"\n**[{q.opportunity}]** ({q.theme} · sentiment={q.sentiment:+.1f})"
            )
            lines.append(f"> \"{q.text}\"")
            lines.append(f"*{q.insight}*")

        lines += ["", "## OST Updates"]
        opp_groups: dict[str, list[Quote]] = {}
        for q in user_quotes:
            opp_groups.setdefault(q.opportunity, []).append(q)
        for opp, quotes in sorted(opp_groups.items(), key=lambda x: -len(x[1])):
            sentiment = sum(q.sentiment for q in quotes) / len(quotes)
            lines.append(f"- **{opp}** — {len(quotes)} signals, avg sentiment={sentiment:+.1f}")

        lines += ["", f"## Next Interview Guide", f"**Target:** {guide.target_profile}"]
        if guide.hypothesis:
            lines.append(f"**Hypothesis to test:** {guide.hypothesis}")
        if guide.warm_up:
            lines += ["", "**Warm-up:**"]
            for w in guide.warm_up[:2]:
                lines.append(f"- {w}")
        lines += ["", "**Questions:**"]
        for i, q in enumerate(guide.questions, 1):
            lines.append(f"\n{i}. {q.get('question', '')}")
            lines.append(f"   *{q.get('rationale', '')}* [{q.get('type', '')}]")
        if guide.ost_gaps:
            lines += ["", "**OST gaps this guide targets:**"]
            for g in guide.ost_gaps:
                lines.append(f"- {g}")
    else:
        lines = [
            f"# Отчёт интервью: {session.participant}",
            f"Продукт: {session.product} | Дата: {session.date} | Сессия: {session.session_id}",
            "",
            "## Ключевые инсайты",
        ]
        for i in session.key_insights:
            lines.append(f"- {i}")

        lines += ["", "## Оспоренные допущения"]
        for a in session.assumptions:
            lines.append(f"- {a}")

        user_quotes = [q for q in session.quotes if q.is_user_quote]
        lines += ["", f"## Цитаты ({len(user_quotes)} от пользователя)"]
        for q in user_quotes[:15]:
            lines.append(
                f"\n**[{q.opportunity}]** ({q.theme} · тональность={q.sentiment:+.1f})"
            )
            lines.append(f"> \"{q.text}\"")
            lines.append(f"*{q.insight}*")

        lines += ["", "## Обновления OST"]
        opp_groups: dict[str, list[Quote]] = {}
        for q in user_quotes:
            opp_groups.setdefault(q.opportunity, []).append(q)
        for opp, quotes in sorted(opp_groups.items(), key=lambda x: -len(x[1])):
            sentiment = sum(q.sentiment for q in quotes) / len(quotes)
            lines.append(f"- **{opp}** — {len(quotes)} сигналов, тональность={sentiment:+.1f}")

        lines += ["", "## Guide для следующего интервью",
                  f"**Профиль:** {guide.target_profile}"]
        if guide.hypothesis:
            lines.append(f"**Гипотеза:** {guide.hypothesis}")
        lines += ["", "**Вопросы:**"]
        for i, q in enumerate(guide.questions, 1):
            lines.append(f"\n{i}. {q.get('question', '')}")
            lines.append(f"   *{q.get('rationale', '')}* [{q.get('type', '')}]")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run(
    source:      TranscriptSource,
    participant: str = "",
    product:     str = "",
    session_id:  str | None = None,
) -> tuple[InterviewSession, InterviewGuide]:
    """
    Full pipeline:
      load → extract (LLM 1) → build session → save → push to CR
      → generate guide (LLM 2) → save guide → write report
    """
    init_db()
    lang = get_language()
    sid  = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 60}")
    print(f"🎤 {L('User Interview Agent', 'Агент интервью')} | "
          f"{L('Session', 'Сессия')}: {sid}")

    stats = db_stats()
    print(f"   {stats['sessions']} {L('sessions', 'сессий')} | "
          f"{stats['quotes']} {L('quotes', 'цитат')} | "
          f"{stats['opportunities']} {L('opportunities in OST', 'возможностей в OST')}")

    # ── Load ─────────────────────────────────
    print(f"\n📄 {L('Loading transcript...', 'Загружаю транскрипт...')}")
    transcript = load_transcript(source)
    print(f"   ✅ {len(transcript):,} {L('chars loaded', 'символов загружено')} ({source.name})")

    # ── Extract (LLM call 1) ──────────────────
    print(f"\n🔍 {L('Extracting quotes + mapping to OST...', 'Извлекаю цитаты + маппю на OST...')}")
    extracted = extract_quotes(transcript, participant, product, lang)

    n_quotes = len(extracted.get("quotes", []))
    n_opps   = len(extracted.get("top_opportunities", []))
    print(f"   ✅ {n_quotes} {L('quotes', 'цитат')} | "
          f"{n_opps} {L('opportunities', 'возможностей')}")

    if extracted.get("participant_profile") and not participant:
        participant = extracted["participant_profile"][:60]

    # ── Build session ─────────────────────────
    session = build_session(sid, participant or "Unknown", product or "Unknown",
                            lang, transcript, extracted)

    # Print top insights
    for insight in session.key_insights[:3]:
        print(f"   💡 {insight[:70]}")

    # ── Save + push to CR ────────────────────
    save_session(session)
    cr_count = push_to_cr(session)
    print(f"\n💾 {L('Saved', 'Сохранено')}: "
          f"{len(session.quotes)} {L('quotes', 'цитат')} | "
          f"CR: {cr_count} {L('signals', 'сигналов')}")

    # ── Generate guide (LLM call 2) ──────────
    print(f"\n📋 {L('Generating next interview guide...', 'Генерирую guide для следующего интервью...')}")
    guide = generate_guide(sid, [sid])
    save_guide(guide)

    n_q = len(guide.questions)
    print(f"   ✅ {n_q} {L('questions', 'вопросов')} | "
          f"Hypothesis: {guide.hypothesis[:60] if guide.hypothesis else '—'}...")

    # ── Report ───────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"interview_report_{sid}_{ts}.md"
    Path(report_file).write_text(
        render_session_report(session, guide), encoding="utf-8"
    )

    print(f"\n{'=' * 60}")
    print(f"✅ {L('Complete', 'Завершено')} | "
          f"{len(session.quotes)} {L('quotes', 'цитат')} → OST | "
          f"CR: {cr_count}")
    print(f"📄 {report_file}")

    return session, guide


# ─────────────────────────────────────────────
# INTERACTIVE INPUT
# ─────────────────────────────────────────────

def collect_input() -> tuple[TranscriptSource, str, str]:
    participant = input(
        L("Participant name/role (e.g. 'Maria, PM at Fintech startup'): ",
          "Имя/роль участника (например 'Мария, PM в финтех стартапе'): ")
    ).strip() or "Unknown"

    product = input(
        L("Product being researched: ",
          "Продукт для исследования: ")
    ).strip() or "Unknown"

    print(L(
        "\nTranscript source:\n  1. Paste text\n  2. Load from file\n  3. Load from URL",
        "\nИсточник транскрипта:\n  1. Вставить текст\n  2. Загрузить из файла\n  3. Загрузить по URL"
    ))
    choice = input(L("Choice (1-3): ", "Выбор (1-3): ")).strip()

    if choice == "2":
        path = input(L("File path: ", "Путь к файлу: ")).strip()
        return FileSource(path), participant, product
    elif choice == "3":
        url = input("URL: ").strip()
        return URLSource(url), participant, product
    else:
        print(L(
            "\nPaste transcript below. Enter blank line twice to finish:",
            "\nВставь транскрипт ниже. Дважды пустая строка — завершить:"
        ))
        lines, blanks = [], 0
        while blanks < 2:
            line = input()
            if not line.strip():
                blanks += 1
            else:
                blanks = 0
            lines.append(line)
        return TextSource("\n".join(lines).strip()), participant, product


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="User Interview Agent")
    p.add_argument("--lang",        choices=["en", "ru"])
    p.add_argument("--file",        default="", help="Path to transcript file")
    p.add_argument("--url",         default="", help="URL to transcript")
    p.add_argument("--participant", default="")
    p.add_argument("--product",     default="")
    p.add_argument("--session",     default=None)
    args = p.parse_args()

    if args.lang:
        set_language(args.lang)

    if args.file:
        source = FileSource(args.file)
    elif args.url:
        source = URLSource(args.url)
    else:
        source, participant, product = collect_input()
        if not args.participant:
            args.participant = participant
        if not args.product:
            args.product = product

    run(source=source,
        participant=args.participant,
        product=args.product,
        session_id=args.session)


if __name__ == "__main__":
    main()
