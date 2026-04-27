"""
Test suite for User Interview Agent.
Run: pytest test_interview_agent.py -v
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test")
sys.path.insert(0, "/mnt/user-data/outputs")
sys.path.insert(0, "/home/claude/interview-agent")

import interview_agent as ia
import i18n

SAMPLE_TRANSCRIPT = """
Interviewer: Tell me about the last time you tried to export your data.
User: Oh god, it was a nightmare. I spent like 40 minutes just trying to find where the export button even was. 
Interviewer: What were you trying to do?
User: I needed to get my monthly report into Excel for my manager. We have this whole process but the tool makes it so hard.
Interviewer: Walk me through exactly what happened.
User: I clicked through every menu, finally found it buried under Settings > Account > Data > Export. Like, why is it there? It should be on the main dashboard.
Interviewer: How did that make you feel?
User: Frustrated. I almost gave up and just did it manually. The whole point of using this tool is to save time.
Interviewer: What would have made it easier?
User: Just put it on the home screen. Or at least let me search for it. Every other tool I use has a search bar for functions.
"""


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ia, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(ia, "CR_DB",   str(tmp_path / "cr.db"))
    ia.init_db()
    # Pre-create CR DB
    conn = sqlite3.connect(str(tmp_path / "cr.db"))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY, source TEXT, raw_text TEXT,
            opportunity TEXT, sentiment REAL, tags_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS opportunities (
            id INTEGER PRIMARY KEY, title TEXT UNIQUE,
            outcome TEXT DEFAULT '', signal_count INTEGER DEFAULT 1,
            avg_sentiment REAL, status TEXT DEFAULT 'open',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()
    yield


@pytest.fixture
def sample_session() -> ia.InterviewSession:
    quotes = [
        ia.Quote(text="I spent 40 minutes finding the export button",
                 speaker="user", opportunity="Export feature discoverability",
                 theme="export", sentiment=-0.7, insight="Critical UX failure",
                 assumption="Export is discoverable", timestamp="00:02:15"),
        ia.Quote(text="It should be on the main dashboard",
                 speaker="user", opportunity="Dashboard information architecture",
                 theme="navigation", sentiment=-0.5, insight="User expects core actions on home screen",
                 assumption="Current IA matches user mental model", timestamp="00:04:10"),
        ia.Quote(text="Let me search for it",
                 speaker="user", opportunity="Command palette / global search",
                 theme="search", sentiment=-0.3, insight="Users want discoverability via search",
                 assumption="Menus are sufficient for navigation", timestamp="00:05:30"),
    ]
    return ia.InterviewSession(
        session_id="test_session",
        participant="Maria, PM at Fintech",
        product="Analytics Dashboard",
        date="2026-04-24",
        raw_text=SAMPLE_TRANSCRIPT,
        lang="en",
        quotes=quotes,
        key_insights=["Export is buried 4 levels deep", "Users expect search for functions"],
        assumptions=["Export IA matches user mental model"],
    )


@pytest.fixture
def sample_guide() -> ia.InterviewGuide:
    return ia.InterviewGuide(
        session_id="test_session",
        based_on=["test_session"],
        ost_gaps=["Export discoverability", "Search functionality"],
        questions=[
            {"question": "Walk me through the last time you needed to share data with someone.",
             "rationale": "Reveals export workflow in context",
             "type": "story"},
            {"question": "What does a typical reporting day look like for you?",
             "rationale": "Uncovers frequency and urgency",
             "type": "exploration"},
        ],
        warm_up=["Tell me a bit about your role and how you use analytics tools day to day."],
        hypothesis="Users need export to be accessible from the main dashboard",
        target_profile="Data-driven PMs or analysts who export weekly",
    )


# ─────────────────────────────────────────────
# 1. DATA MODELS
# ─────────────────────────────────────────────

class TestDataModels:
    def test_quote_is_user_quote(self):
        q = ia.Quote(text="test", speaker="user", opportunity="O",
                     theme="t", sentiment=0.0, insight="i", assumption="a")
        assert q.is_user_quote is True

    def test_quote_interviewer_not_user(self):
        q = ia.Quote(text="test", speaker="interviewer", opportunity="O",
                     theme="t", sentiment=0.0, insight="i", assumption="a")
        assert q.is_user_quote is False

    def test_quote_participant_is_user(self):
        q = ia.Quote(text="test", speaker="participant", opportunity="O",
                     theme="t", sentiment=0.0, insight="i", assumption="a")
        assert q.is_user_quote is True

    def test_session_defaults(self):
        session = ia.InterviewSession(
            session_id="s1", participant="Test", product="Prod",
            date="2026-01-01", raw_text="text", lang="en",
        )
        assert session.quotes == []
        assert session.key_insights == []
        assert session.assumptions == []
        assert session.created_at != ""

    def test_guide_defaults(self):
        guide = ia.InterviewGuide(
            session_id="s1", based_on=["s1"],
            ost_gaps=["gap1"], questions=[],
            warm_up=[], hypothesis="", target_profile="",
        )
        assert guide.warm_up == []
        assert guide.hypothesis == ""
        assert guide.target_profile == ""
        assert guide.created_at != ""


# ─────────────────────────────────────────────
# 2. TRANSCRIPT SOURCES
# ─────────────────────────────────────────────

class TestTranscriptSources:
    def test_text_source_loads(self):
        src = ia.TextSource(SAMPLE_TRANSCRIPT)
        text = src.load()
        assert "export button" in text
        assert src.name == "text"

    def test_text_source_strips(self):
        src = ia.TextSource("   hello   ")
        assert src.load() == "hello"

    def test_file_source_loads(self, tmp_path):
        f = tmp_path / "transcript.txt"
        f.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
        src = ia.FileSource(str(f))
        text = src.load()
        assert "export button" in text
        assert src.name == "file"

    def test_file_source_missing(self, tmp_path):
        src = ia.FileSource(str(tmp_path / "nonexistent.txt"))
        with pytest.raises(FileNotFoundError):
            src.load()

    def test_url_source_name(self):
        src = ia.URLSource("https://example.com")
        assert src.name == "url"

    def test_transcript_too_short_raises(self):
        src = ia.TextSource("hi")
        with pytest.raises(ValueError, match="too short"):
            ia.load_transcript(src)

    def test_load_transcript_cleans(self):
        raw = "  Line 1\r\nLine 2\r\n\r\n\r\nLine 3  "
        src = ia.TextSource(raw)
        text = ia.load_transcript(src)
        assert "\r" not in text
        assert not text.startswith(" ")


# ─────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────

class TestHelpers:
    def test_L_en(self):
        i18n.set_language("en")
        assert ia.L("Hello", "Привет") == "Hello"

    def test_L_ru(self):
        i18n.set_language("ru")
        assert ia.L("Hello", "Привет") == "Привет"

    def test_clean_transcript_removes_crlf(self):
        text = "line1\r\nline2\r\nline3"
        cleaned = ia._clean_transcript(text)
        assert "\r" not in cleaned

    def test_clean_transcript_collapses_blanks(self):
        text = "line1\n\n\n\n\nline2"
        cleaned = ia._clean_transcript(text)
        assert "\n\n\n" not in cleaned

    def test_truncate_short(self):
        text = "short text"
        assert ia._truncate(text, 1000) == text

    def test_truncate_long(self):
        text = "a" * 30000
        result = ia._truncate(text, 1000)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_truncate_keeps_beginning_and_end(self):
        text = "START" + "x" * 30000 + "END"
        result = ia._truncate(text, 1000)
        assert "START" in result
        assert "END" in result


# ─────────────────────────────────────────────
# 4. DATABASE
# ─────────────────────────────────────────────

class TestDatabase:
    def test_init_creates_tables(self, tmp_db):
        conn = sqlite3.connect(ia.DB_PATH)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert {"sessions", "quotes", "guides"} <= tables

    def test_save_session_persists(self, tmp_db, sample_session):
        ia.save_session(sample_session)
        conn = sqlite3.connect(ia.DB_PATH)
        row = conn.execute(
            "SELECT participant, product FROM sessions WHERE session_id=?",
            ("test_session",)
        ).fetchone()
        conn.close()
        assert row[0] == "Maria, PM at Fintech"
        assert row[1] == "Analytics Dashboard"

    def test_save_session_saves_quotes(self, tmp_db, sample_session):
        ia.save_session(sample_session)
        conn = sqlite3.connect(ia.DB_PATH)
        n = conn.execute(
            "SELECT COUNT(*) FROM quotes WHERE session_id='test_session'"
        ).fetchone()[0]
        conn.close()
        assert n == 3

    def test_save_session_idempotent(self, tmp_db, sample_session):
        ia.save_session(sample_session)
        ia.save_session(sample_session)  # OR REPLACE
        conn = sqlite3.connect(ia.DB_PATH)
        n = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        conn.close()
        assert n == 1

    def test_save_guide(self, tmp_db, sample_guide):
        ia.save_guide(sample_guide)
        conn = sqlite3.connect(ia.DB_PATH)
        row = conn.execute("SELECT hypothesis FROM guides").fetchone()
        conn.close()
        assert "export" in row[0].lower()

    def test_get_past_sessions_empty(self, tmp_db):
        assert ia.get_past_sessions() == []

    def test_get_past_sessions_returns_latest(self, tmp_db, sample_session):
        ia.save_session(sample_session)
        sessions = ia.get_past_sessions()
        assert len(sessions) == 1
        assert sessions[0]["participant"] == "Maria, PM at Fintech"

    def test_get_ost_opportunities_sorted(self, tmp_db, sample_session):
        ia.save_session(sample_session)
        opps = ia.get_ost_opportunities()
        assert len(opps) > 0
        # sorted by quote_count desc
        if len(opps) > 1:
            assert opps[0]["quote_count"] >= opps[1]["quote_count"]

    def test_db_stats_structure(self, tmp_db):
        stats = ia.db_stats()
        for key in ["sessions", "quotes", "guides", "opportunities"]:
            assert key in stats
            assert isinstance(stats[key], int)

    def test_db_stats_increments(self, tmp_db, sample_session):
        ia.save_session(sample_session)
        stats = ia.db_stats()
        assert stats["sessions"] == 1
        assert stats["quotes"] == 3
        assert stats["opportunities"] == 3


# ─────────────────────────────────────────────
# 5. CR INTEGRATION
# ─────────────────────────────────────────────

class TestCRIntegration:
    def test_push_to_cr_saves_user_quotes(self, tmp_db, sample_session):
        n = ia.push_to_cr(sample_session)
        assert n == 3  # all 3 quotes are from "user"

    def test_push_to_cr_creates_signals(self, tmp_db, sample_session):
        ia.push_to_cr(sample_session)
        conn = sqlite3.connect(ia.CR_DB)
        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        conn.close()
        assert count == 3

    def test_push_to_cr_creates_opportunities(self, tmp_db, sample_session):
        ia.push_to_cr(sample_session)
        conn = sqlite3.connect(ia.CR_DB)
        count = conn.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]
        conn.close()
        assert count == 3

    def test_push_to_cr_source_is_user_interview(self, tmp_db, sample_session):
        ia.push_to_cr(sample_session)
        conn = sqlite3.connect(ia.CR_DB)
        sources = conn.execute("SELECT DISTINCT source FROM signals").fetchall()
        conn.close()
        assert all(r[0] == "user_interview" for r in sources)

    def test_push_to_cr_skips_interviewer_quotes(self, tmp_db):
        session = ia.InterviewSession(
            session_id="s2", participant="Test", product="P",
            date="2026-01-01", raw_text="t", lang="en",
            quotes=[
                ia.Quote(text="What do you think about X?", speaker="interviewer",
                         opportunity="O", theme="t", sentiment=0.0, insight="i", assumption="a"),
            ]
        )
        n = ia.push_to_cr(session)
        assert n == 0

    def test_push_to_cr_empty_quotes(self, tmp_db):
        session = ia.InterviewSession(
            session_id="s3", participant="Test", product="P",
            date="2026-01-01", raw_text="t", lang="en",
        )
        assert ia.push_to_cr(session) == 0


# ─────────────────────────────────────────────
# 6. BUILD SESSION
# ─────────────────────────────────────────────

class TestBuildSession:
    def test_build_session_filters_short_quotes(self, tmp_db):
        extracted = {
            "quotes": [
                {"text": "OK", "speaker": "user", "opportunity": "O",
                 "theme": "t", "sentiment": 0.0, "insight": "i",
                 "assumption": "a", "timestamp": ""},
                {"text": "This is a proper quote about the feature",
                 "speaker": "user", "opportunity": "Feature usability",
                 "theme": "ux", "sentiment": -0.5, "insight": "UX issue",
                 "assumption": "Feature is usable", "timestamp": "00:01:00"},
            ],
            "key_insights": ["Insight 1"],
            "assumptions_challenged": ["Assumption 1"],
            "top_opportunities": ["Feature usability"],
        }
        session = ia.build_session("s1", "Test", "Prod", "en",
                                   SAMPLE_TRANSCRIPT, extracted)
        assert len(session.quotes) == 1
        assert session.quotes[0].opportunity == "Feature usability"

    def test_build_session_preserves_insights(self, tmp_db):
        extracted = {
            "quotes": [],
            "key_insights": ["Insight A", "Insight B"],
            "assumptions_challenged": ["Assumption X"],
            "top_opportunities": [],
        }
        session = ia.build_session("s2", "Test", "Prod", "en",
                                   SAMPLE_TRANSCRIPT, extracted)
        assert "Insight A" in session.key_insights
        assert "Assumption X" in session.assumptions

    def test_build_session_handles_missing_fields(self, tmp_db):
        session = ia.build_session("s3", "Test", "Prod", "en",
                                   SAMPLE_TRANSCRIPT, {})
        assert session.quotes == []
        assert session.key_insights == []


# ─────────────────────────────────────────────
# 7. REPORT RENDERING
# ─────────────────────────────────────────────

class TestReportRendering:
    def test_report_en_structure(self, tmp_db, sample_session, sample_guide):
        i18n.set_language("en")
        report = ia.render_session_report(sample_session, sample_guide)
        assert "Interview Report" in report
        assert "Key Insights" in report
        assert "Next Interview Guide" in report
        assert sample_session.participant in report

    def test_report_ru_structure(self, tmp_db, sample_session, sample_guide):
        i18n.set_language("ru")
        report = ia.render_session_report(sample_session, sample_guide)
        assert "Отчёт интервью" in report
        assert "Ключевые инсайты" in report
        assert "Guide для следующего интервью" in report

    def test_report_contains_quotes(self, tmp_db, sample_session, sample_guide):
        i18n.set_language("en")
        report = ia.render_session_report(sample_session, sample_guide)
        assert "export button" in report.lower()

    def test_report_contains_guide_questions(self, tmp_db, sample_session, sample_guide):
        i18n.set_language("en")
        report = ia.render_session_report(sample_session, sample_guide)
        assert "Walk me through" in report

    def test_report_contains_ost_updates(self, tmp_db, sample_session, sample_guide):
        i18n.set_language("en")
        report = ia.render_session_report(sample_session, sample_guide)
        assert "OST" in report


# ─────────────────────────────────────────────
# 8. PIPELINE (mocked LLM)
# ─────────────────────────────────────────────

class TestPipeline:
    def _mock_call(self, monkeypatch, is_extract: bool):
        if is_extract:
            rv = {
                "quotes": [
                    {"text": "I spent 40 minutes finding the export button",
                     "speaker": "user", "opportunity": "Export discoverability",
                     "theme": "export", "sentiment": -0.7,
                     "insight": "Export is critically hard to find",
                     "assumption": "Export is discoverable", "timestamp": "00:02:15"},
                    {"text": "It should be on the main dashboard",
                     "speaker": "user", "opportunity": "Dashboard IA",
                     "theme": "navigation", "sentiment": -0.5,
                     "insight": "Users expect export on home", "assumption": "IA is correct",
                     "timestamp": ""},
                ],
                "key_insights": ["Export is buried 4 levels deep"],
                "assumptions_challenged": ["Export IA matches user mental model"],
                "top_opportunities": ["Export discoverability", "Dashboard IA"],
                "participant_profile": "Data analyst, exports weekly",
            }
        else:
            rv = {
                "target_profile": "PMs who export data weekly",
                "hypothesis": "Export should be on main dashboard",
                "warm_up": ["Tell me about your reporting workflow."],
                "questions": [
                    {"question": "Walk me through the last time you shared data.",
                     "rationale": "Reveals export context", "type": "story"},
                ],
                "probes": ["What happened next?"],
                "things_to_avoid": ["Asking about specific features"],
            }
        return rv

    def test_pipeline_runs_end_to_end(self, tmp_db, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        call_count = [0]
        def mock_call(system, user):
            is_extract = "quotes" not in system.lower() or call_count[0] == 0
            call_count[0] += 1
            return self._mock_call(monkeypatch, call_count[0] == 1)
        monkeypatch.setattr(ia, "_call", mock_call)

        src = ia.TextSource(SAMPLE_TRANSCRIPT)
        session, guide = ia.run(src, "Maria", "Analytics")

        assert isinstance(session, ia.InterviewSession)
        assert isinstance(guide, ia.InterviewGuide)
        assert len(session.quotes) > 0

    def test_pipeline_saves_to_db(self, tmp_db, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        call_count = [0]
        def mock_call(system, user):
            call_count[0] += 1
            return self._mock_call(monkeypatch, call_count[0] == 1)
        monkeypatch.setattr(ia, "_call", mock_call)

        src = ia.TextSource(SAMPLE_TRANSCRIPT)
        ia.run(src, "Test", "Prod")

        stats = ia.db_stats()
        assert stats["sessions"] == 1
        assert stats["quotes"] >= 1
        assert stats["guides"] == 1

    def test_pipeline_pushes_to_cr(self, tmp_db, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        call_count = [0]
        def mock_call(system, user):
            call_count[0] += 1
            return self._mock_call(monkeypatch, call_count[0] == 1)
        monkeypatch.setattr(ia, "_call", mock_call)

        src = ia.TextSource(SAMPLE_TRANSCRIPT)
        ia.run(src, "Test", "Prod")

        conn = sqlite3.connect(ia.CR_DB)
        n = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        conn.close()
        assert n >= 1

    def test_pipeline_creates_report_file(self, tmp_db, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        call_count = [0]
        def mock_call(system, user):
            call_count[0] += 1
            return self._mock_call(monkeypatch, call_count[0] == 1)
        monkeypatch.setattr(ia, "_call", mock_call)

        src = ia.TextSource(SAMPLE_TRANSCRIPT)
        ia.run(src, "Test", "Prod")

        reports = list(tmp_path.glob("interview_report_*.md"))
        assert len(reports) == 1

    def test_pipeline_lm_calls_count(self, tmp_db, monkeypatch, tmp_path):
        """Verify exactly 2 LLM calls — not more."""
        monkeypatch.chdir(tmp_path)
        call_count = [0]
        def mock_call(system, user):
            call_count[0] += 1
            return self._mock_call(monkeypatch, call_count[0] == 1)
        monkeypatch.setattr(ia, "_call", mock_call)

        src = ia.TextSource(SAMPLE_TRANSCRIPT)
        ia.run(src, "Test", "Prod")

        assert call_count[0] == 2  # extract + guide

    def test_pipeline_file_source(self, tmp_db, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "transcript.txt"
        f.write_text(SAMPLE_TRANSCRIPT)
        call_count = [0]
        def mock_call(system, user):
            call_count[0] += 1
            return self._mock_call(monkeypatch, call_count[0] == 1)
        monkeypatch.setattr(ia, "_call", mock_call)

        src = ia.FileSource(str(f))
        session, _ = ia.run(src, "Test", "Prod")
        assert session.session_id != ""


# ─────────────────────────────────────────────
# 9. I18N
# ─────────────────────────────────────────────

class TestI18n:
    def test_prompts_have_lang_placeholder(self):
        assert "{lang}" in ia.EXTRACT_PROMPT
        assert "{lang}" in ia.GUIDE_PROMPT

    def test_language_instruction_en(self):
        i18n.set_language("en")
        assert "English" in i18n.get_language_instruction()

    def test_language_instruction_ru(self):
        i18n.set_language("ru")
        assert "русском" in i18n.get_language_instruction()

    def test_prompt_finalize_both_langs(self):
        for lang in ["en", "ru"]:
            i18n.set_language(lang)
            val = i18n.t("prompt_finalize")
            assert isinstance(val, str) and len(val) > 0
