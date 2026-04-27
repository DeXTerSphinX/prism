# Prism — PR Compliance Checker

> Built for the SimplifAI Hackathon (AI Engineering Track) · Team 1–4

Prism automates pull request reviews by checking whether code changes actually satisfy the requirements in a Jira ticket. Paste a Jira ticket and a GitHub PR URL — Prism fetches the diff, extracts requirements using an LLM, checks each one against the code, and returns a compliance score with per-requirement verdicts.

---

## The Problem

Manual PR reviews are slow and inconsistent. Reviewers check for style and logic, but rarely verify systematically whether every requirement from the ticket was actually implemented. Requirement drift goes undetected until QA or production.

---

## What Prism Does

1. Accepts a Jira ticket (pasted as text) and a GitHub PR URL
2. Fetches the PR diff via the GitHub REST API
3. Extracts discrete requirements from the ticket using an LLM
4. Checks each requirement against the diff and assigns a verdict: **PASS**, **PARTIAL**, or **FAIL**
5. Computes an overall compliance score (0–100)
6. Generates a natural-language summary of the review

---

## Demo

> **Live demo:** https://prism-aduk.onrender.com — hosted on Render free tier, may take ~30 seconds to wake up on first load.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | Python, Flask |
| LLM | Groq API (`llama-3.1-8b-instant`) |
| GitHub Integration | GitHub REST API (PR diff fetching) |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Render (Procfile included) |

**Note on stack deviation:** The hackathon problem statement specified MCP servers and AI agents as the required stack. I chose direct API calls instead — the pipeline is sequential and deterministic, making an agentic multi-step approach unnecessarily complex for this scope. The trade-off is less flexibility but simpler, more debuggable code.

---

## Pipeline

```
Jira Ticket Text + GitHub PR URL
        │
        ▼
  fetch_pr_diff()      → GitHub REST API → raw unified diff
        │
        ▼
  filter_diff()        → keeps +/- lines and @@ headers, drops noise
        │
        ▼
  extract_requirements() → LLM → list of discrete requirements
        │
        ▼
  analyze_requirements() → LLM → per-requirement verdict + evidence
        │
        ▼
  compute_compliance_score() → integer 0–100
        │
        ▼
  generate_review_summary()  → LLM → 1-2 sentence natural language summary
```

---

## API

### `POST /analyze`

**Request body:**
```json
{
  "jira_ticket": "As a user, I want...",
  "pr_url": "https://github.com/owner/repo/pull/123"
}
```

**Response:**
```json
{
  "compliance_score": 75,
  "summary": "The PR implements authentication and input validation but is missing the rate limiting requirement.",
  "results": [
    {
      "requirement": "Add JWT authentication to the /login endpoint",
      "verdict": "PASS",
      "evidence": "+ token = jwt.encode(payload, SECRET_KEY)"
    },
    {
      "requirement": "Add rate limiting to prevent brute force",
      "verdict": "FAIL",
      "evidence": "No evidence found"
    }
  ]
}
```

### `GET /health`
Returns `{"status": "ok"}` — used for deployment health checks.

---

## Local Setup

```bash
git clone https://github.com/DeXTerSphinX/prism.git
cd prism

pip install -r requirements.txt
```

Create a `.env` file:
```
GITHUB_TOKEN=your_github_token
GROQ_API_KEY=your_groq_api_key
```

Run:
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Known Limitations

- Diff is truncated at 4000 characters to stay within LLM context limits — very large PRs may lose tail context
- LLM verdicts are probabilistic, not deterministic — edge cases may produce inconsistent results
- No authentication on the `/analyze` endpoint
- Jira ticket must be pasted as plain text — no Jira API integration

---

## Hackathon Context

Built during the SimplifAI Hackathon (AI Engineering Track). The problem statement and judging criteria were provided by SimplifAI/Prospire Technology Services. All implementation — architecture, backend, frontend, and prompt design — is original work.
