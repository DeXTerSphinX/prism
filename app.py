import json
import logging
import os
import re
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GITHUB_TOKEN:
    logger.warning("GITHUB_TOKEN is not set — private repos will return 404")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set — LLM calls will fail")


# ---------------------------------------------------------------------------
# Placeholder helpers
# ---------------------------------------------------------------------------

def fetch_pr_diff(pr_url: str) -> str:
    """
    Fetch the unified diff for a GitHub pull request via the GitHub REST API.

    Converts a PR URL such as
        https://github.com/owner/repo/pull/123
    into the API call:
        GET https://api.github.com/repos/owner/repo/pulls/123
    with Accept: application/vnd.github.v3.diff
    """
    pattern = r"^https://github\.com/([^/]+)/([^/]+)/pull/(\d+)$"
    match = re.match(pattern, pr_url.rstrip("/"))
    if not match:
        raise ValueError(
            "Expected format: https://github.com/owner/repo/pull/123"
        )

    owner, repo, pull_number = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"

    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "User-Agent": "PRISM/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        response = requests.get(api_url, headers=headers, timeout=20)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not contact GitHub: {exc}") from exc

    if response.status_code == 404:
        hint = " (is the repo private? Set GITHUB_TOKEN)" if not GITHUB_TOKEN else " (check the PR URL is correct)"
        raise RuntimeError(f"PR not found{hint}")
    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub returned {response.status_code}: {response.text[:200]}"
        )

    diff = response.text.strip()

    if not diff:
        raise RuntimeError("GitHub returned an empty diff")

    return diff


def filter_diff(diff_text: str, char_limit: int = 4000) -> str:
    """
    Reduce noise in a unified diff before sending it to the LLM.

    Keeps only added lines (starting with "+"), excludes file-header
    lines ("+++"), joins them, and truncates to `char_limit` characters.

    Returns the filtered diff string.
    """
    added_lines = [
        line for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    filtered = "\n".join(added_lines)
    return filtered[:char_limit]


def extract_requirements(jira_text: str) -> list[str]:
    """
    Extract discrete requirements from a Jira ticket using Groq.

    Returns:
        List of requirement strings.

    Raises:
        RuntimeError if the API call fails or parsing fails.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")

    if not jira_text or not jira_text.strip():
        raise RuntimeError("Jira ticket text is empty")

    system_prompt = (
        "You extract software requirements from Jira tickets. "
        "Return ONLY a JSON array of requirement strings. "
        "Do not include explanations or markdown."
    )

    user_prompt = f"Jira ticket:\n{jira_text}"

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 300,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Groq API request timed out")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Groq API request failed: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"Groq API returned {resp.status_code}: {resp.text}"
        )

    try:
        content = resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError):
        raise RuntimeError(f"Unexpected Groq response format: {resp.text}")

    # Remove markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL).strip()

    try:
        requirements = json.loads(cleaned)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse requirements JSON: {content}")

    if not isinstance(requirements, list):
        raise RuntimeError(f"Expected list of requirements, got: {requirements}")

    requirements = [r.strip() for r in requirements if isinstance(r, str) and r.strip()]

    if not requirements:
        raise RuntimeError("No requirements extracted from Jira ticket")

    return requirements


def analyze_requirements(requirements: list[str], diff: str) -> list[dict]:
    """
    Compare each requirement against the PR diff and produce a verdict.

    For every requirement, return a dict with:
        {
            "requirement": str,
            "verdict":     "PASS" | "PARTIAL" | "FAIL",
            "evidence":    str   # relevant code snippet from the diff
        }

    Returns a list of such dicts, one per requirement.
    Raises RuntimeError on API or parse failures.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")

    VALID_VERDICTS = {"PASS", "PARTIAL", "FAIL"}

    numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(requirements))

    system_prompt = (
    "You are a senior software engineer performing a pull request review.\n"
    "You will receive:\n"
    "1. A list of requirements extracted from a Jira ticket.\n"
    "2. A filtered pull request diff containing only added lines.\n\n"

    "Your task is to determine whether each requirement is implemented in the diff.\n\n"

    "Return ONLY valid JSON. Do not include explanations, markdown, or text outside JSON.\n\n"

    "The JSON format must be exactly:\n"
    '{"results":[{"requirement":"string","verdict":"PASS|PARTIAL|FAIL","evidence":"string"}]}\n\n'

    "Rules:\n"
    "- PASS: requirement clearly implemented.\n"
    "- PARTIAL: partially implemented or unclear.\n"
    "- FAIL: no evidence found in the diff.\n"
    "- evidence must be a short code snippet from the diff or 'No evidence found'."
)

    user_prompt = (
        f"Requirements:\n{numbered}\n\n"
        f"Filtered diff (added lines):\n{diff}"
    )

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 600
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("Groq API request timed out")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Groq API returned {response.status_code}: {response.reason}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Groq API request failed: {exc}") from exc

    raw_content = response.json()["choices"][0]["message"]["content"].strip()

    # Parse JSON — strip markdown fences and retry once on failure
    def parse_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Failed to parse JSON from model response: {text!r}"
                ) from exc

    parsed = parse_json(raw_content)

    if not isinstance(parsed, dict) or "results" not in parsed:
        raise RuntimeError(
            f"Model response missing 'results' key: {parsed!r}"
        )

    raw_results = parsed["results"]
    if not isinstance(raw_results, list):
        raise RuntimeError(
            f"Expected 'results' to be a list, got: {type(raw_results).__name__}"
        )

    # Validate and sanitize each item — be lenient rather than rejecting everything
    results = []
    for i, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict", "FAIL")).upper().strip()
        if verdict not in VALID_VERDICTS:
            verdict = "FAIL"
        results.append({
            "requirement": str(item.get("requirement", requirements[i] if i < len(requirements) else "")),
            "verdict":     verdict,
            "evidence":    str(item.get("evidence", "No evidence found")),
        })

    return results


def compute_compliance_score(results: list) -> int:
    """
    Compute a PR compliance score from a list of requirement verdicts.

    Scoring:  PASS = 1 pt  |  PARTIAL = 0.5 pts  |  FAIL = 0 pts
    Formula:  round((total_points / total_requirements) * 100)

    Returns an integer between 0 and 100.
    Returns 0 for an empty or malformed results list.
    """
    POINTS = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}

    if not isinstance(results, list) or len(results) == 0:
        return 0

    total_points = 0.0
    total_requirements = 0

    for item in results:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict", "")).upper().strip()
        total_points += POINTS.get(verdict, 0.0)
        total_requirements += 1

    if total_requirements == 0:
        return 0

    return round((total_points / total_requirements) * 100)


def generate_review_summary(requirements: list[str], results: list[dict]) -> str:
    """
    Generate a concise 1-2 sentence natural-language summary of the PR review.

    Uses the requirement list and verdict results to ask the LLM to describe
    the overall implementation quality.

    Returns a plain-text summary string.
    Falls back to a rule-based summary if the LLM call fails for any reason.
    """
    def _fallback_summary() -> str:
        if not results:
            return "No requirements were evaluated."
        counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
        for r in results:
            counts[r.get("verdict", "FAIL")] = counts.get(r.get("verdict", "FAIL"), 0) + 1
        total = len(results)
        if counts["PASS"] == total:
            return "The PR fully implements all Jira requirements."
        if counts["FAIL"] == total:
            return "The PR does not appear to implement any of the Jira requirements."
        return (
            f"The PR implements {counts['PASS']} of {total} requirements fully, "
            f"with {counts['PARTIAL']} partial and {counts['FAIL']} missing."
        )

    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — using fallback summary")
        return _fallback_summary()

    verdict_lines = "\n".join(
        f"- [{r.get('verdict', 'FAIL')}] {r.get('requirement', '')}"
        for r in results
    )

    system_prompt = (
        "You are a senior code reviewer writing a pull request summary. "
        "Given a list of requirements and their verdicts, write a concise 1-2 sentence "
        "summary describing the overall PR quality. "
        "Be specific: mention what was implemented well and what is missing. "
        "Return only the summary text — no bullet points, no headings, no markdown."
    )

    user_prompt = (
        f"Requirements and verdicts:\n{verdict_lines}\n\n"
        "Write a short summary of the PR's implementation quality."
    )

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 120,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"].strip()
        if not summary:
            raise ValueError("Empty summary returned by model")
        return summary
    except Exception as exc:
        logger.warning("Summary generation failed (%s) — using fallback", exc)
        return _fallback_summary()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(".", "prism.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    # ── 1. Parse and validate request body ───────────────────────────────
    body = request.get_json(silent=True)
    if not body:
        logger.warning("Rejected request: missing or non-JSON body")
        return jsonify({"error": "Request body must be JSON"}), 400

    jira_ticket = body.get("jira_ticket", "").strip()
    pr_url      = body.get("pr_url", "").strip()

    if not jira_ticket:
        logger.warning("Rejected request: missing jira_ticket")
        return jsonify({"error": "Missing required field: jira_ticket"}), 400
    if not pr_url:
        logger.warning("Rejected request: missing pr_url")
        return jsonify({"error": "Missing required field: pr_url"}), 400

    logger.info("Received analysis request for PR: %s", pr_url)

    # ── 2. Fetch PR diff ──────────────────────────────────────────────────
    logger.info("Fetching PR diff...")
    try:
        raw_diff = fetch_pr_diff(pr_url)
    except ValueError as exc:
        logger.warning("Invalid PR URL '%s': %s", pr_url, exc)
        return jsonify({"error": f"Invalid PR URL: {exc}"}), 400
    except RuntimeError as exc:
        logger.error("Failed to fetch PR diff: %s", exc)
        return jsonify({"error": f"Failed to fetch PR diff: {exc}"}), 502

    logger.info("Fetched raw diff (%d chars)", len(raw_diff))

    # ── 3. Filter diff ────────────────────────────────────────────────────
    diff = filter_diff(raw_diff)
    logger.info("Filtered diff to %d chars", len(diff))

    # ── 4. Extract requirements ───────────────────────────────────────────
    logger.info("Extracting requirements from Jira ticket...")
    try:
        requirements = extract_requirements(jira_ticket)
    except RuntimeError as exc:
        logger.error("Failed to extract requirements: %s", exc)
        return jsonify({"error": f"Failed to extract requirements: {exc}"}), 502

    if not requirements:
        logger.warning("No requirements extracted from ticket")
        return jsonify({"error": "No requirements could be extracted from the ticket"}), 422

    logger.info("Extracted %d requirement(s)", len(requirements))

    # ── 5. Analyze requirements against the diff ──────────────────────────
    logger.info("Analyzing requirements against diff...")
    try:
        results = analyze_requirements(requirements, diff)
    except RuntimeError as exc:
        logger.error("Failed to analyze requirements: %s", exc)
        return jsonify({"error": f"Failed to analyze requirements: {exc}"}), 502

    logger.info(
        "Analysis complete — %d result(s): %s",
        len(results),
        {v: sum(1 for r in results if r["verdict"] == v) for v in ("PASS", "PARTIAL", "FAIL")},
    )

    compliance_score = compute_compliance_score(results)
    logger.info("Compliance score: %d%%", compliance_score)

    # ── 6. Generate review summary ────────────────────────────────────────
    logger.info("Generating review summary...")
    summary = generate_review_summary(requirements, results)
    logger.info("Summary: %s", summary)

    # ── 7. Return structured response ─────────────────────────────────────
    return jsonify({
        "compliance_score": compliance_score,
        "summary":          summary,
        "results":          results,
    }), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)

