# 🎤 User Interview Agent

Processes user interview transcripts using Teresa Torres' Continuous Discovery Habits framework. Extracts quotes, maps them to the Opportunity Solution Tree, and generates the next interview guide based on OST gaps.

## What it does

1. **Loads transcript** — from text, file, or URL
2. **Extracts quotes** (LLM call 1) — maps each user quote to an OST opportunity with sentiment, theme, insight, and assumption
3. **Updates OST** — writes user quotes as signals to the CR agent's database
4. **Generates interview guide** (LLM call 2) — targets opportunities with weakest evidence, follows Teresa Torres' questioning principles
5. **Saves report** — markdown with quotes, OST updates, and next guide

Exactly **2 LLM calls** per session — not per quote.

## Quick start

```bash
git clone https://github.com/artemsanisimow-ux/interview-agent.git
cd interview-agent
python3 -m venv venv
source venv/bin/activate
pip install langchain-anthropic python-dotenv requests
```

Add to `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
LANGUAGE=en
```

```bash
# Interactive — paste transcript
python3 interview_agent.py --lang en

# From file
python3 interview_agent.py --lang en --file transcript.txt

# From URL (Notion export, Google Doc, etc.)
python3 interview_agent.py --lang en --url https://notion.so/...
```

## Transcript sources

| Source | How |
|--------|-----|
| Text | Paste directly into terminal |
| File | `--file path/to/transcript.txt` |
| URL | `--url https://...` — fetches and strips HTML |

## Output

- `interview_report_SESSION_TIMESTAMP.md` — quotes, OST updates, next guide
- `interviews.db` — SQLite with sessions, quotes, guides
- CR agent DB — user quotes written as `user_interview` signals automatically

## OST gap detection

The guide generator queries the local DB for opportunities with fewer than 3 supporting quotes and prioritizes those in the next interview. Guides improve automatically as more sessions accumulate.

## Running tests

```bash
pytest test_interview_agent.py -v   # 53 tests
```

Covers: data models, transcript sources, helpers, DB persistence, CR integration, build session, report rendering, pipeline (mocked LLM with call count assertion), i18n.

## Part of a larger system

| Agent | Repo |
|-------|------|
| Discovery | [discovery-agent](https://github.com/artemsanisimow-ux/discovery-agent) |
| Grooming | [grooming-agent](https://github.com/artemsanisimow-ux/grooming-agent) |
| Planning | [planning-agent](https://github.com/artemsanisimow-ux/planning-agent) |
| Retrospective | [retro-agent](https://github.com/artemsanisimow-ux/retro-agent) |
| PRD | [prd-agent](https://github.com/artemsanisimow-ux/prd-agent) |
| Stakeholder | [stakeholder-agent](https://github.com/artemsanisimow-ux/stakeholder-agent) |
| A/B Testing | [ab-agent](https://github.com/artemsanisimow-ux/ab-agent) |
| Continuous Research | [cr-agent](https://github.com/artemsanisimow-ux/cr-agent) |
| Orchestrator | [orchestrator](https://github.com/artemsanisimow-ux/orchestrator) |
| Metrics Monitor | [metrics-agent](https://github.com/artemsanisimow-ux/metrics-agent) |
| User Interview | this repo |
