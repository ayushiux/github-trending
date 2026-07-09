"""
GitHub Trending AI/LLM Tracker — Daily Personalized Digest.

Scrapes github.com/trending, uses an LLM to figure out what each repo actually is
and scores it against per-person profiles (profiles/*.md — Sumeet, Ayushi), tracks
novelty (new / surging / recurring) across days, renders a scannable HTML brief,
emails it, and publishes it to GitHub Pages for a phone-openable link.

Usage:
    python trending.py              # Fetch, analyze, email + publish
    python trending.py --dry-run    # Save report only; no email, no publish
    python trending.py --no-email   # Skip email (still publishes unless --no-publish)
    python trending.py --no-publish # Skip GitHub Pages publish
    python trending.py --force       # Ignore analysis cache (re-analyze everything)
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Logging — unbuffered for cron/task scheduler
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
STATE_DIR = SCRIPT_DIR / "state"
PROFILES_DIR = SCRIPT_DIR / "profiles"
DOCS_DIR = SCRIPT_DIR / "docs"
ARCHIVE_DIR = DOCS_DIR / "archive"
for d in (OUTPUT_DIR, STATE_DIR, DOCS_DIR, ARCHIVE_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEEN_PATH = STATE_DIR / "seen_repos.json"
ANALYSIS_CACHE_PATH = STATE_DIR / "analysis_cache.json"


# ---------------------------------------------------------------------------
# Local config (gitignored) — keeps personal values out of the public repo
# ---------------------------------------------------------------------------
def _load_local_config() -> dict:
    p = SCRIPT_DIR / "config.local.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not parse config.local.json: {exc}")
    return {}


_CFG = _load_local_config()

EMAIL_TO = os.getenv("GHTREND_EMAIL_TO") or _CFG.get("email_to") or ""
BROWSER_AGENT = Path(
    os.getenv("GHTREND_BROWSER_AGENT")
    or _CFG.get("browser_agent")
    or r"C:\Users\suagraw\Ayushi\browser-agent"
)

# GitHub Pages publish target
PAGES_REPO = _CFG.get("pages_repo", "ayushiux/github-trending")
PAGES_URL = _CFG.get("pages_url", "https://ayushiux.github.io/github-trending/")
PAGES_ACCOUNT = _CFG.get("gh_account", "ayushiux")
PAGES_RESTORE_ACCOUNT = _CFG.get("gh_restore_account", "agrawalsumeet25-dot")
PAGES_BRANCH = _CFG.get("pages_branch", "main")

# LLM (local proxy — same infra as luma pipeline)
LLM_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:5000")
LLM_API_KEY = os.getenv("ANTHROPIC_AUTH_TOKEN", "your-anthropic-auth-token")
LLM_MODEL = "claude-sonnet-4-6"
LLM_MAX_WORKERS = 16
RELEVANCE_THRESHOLD = 55  # max(person scores) >= this => surfaced

CATEGORIES = [
    "Agents", "Coding Tools", "Models", "RAG/Search",
    "MCP", "Infra", "Apps", "Research", "Other",
]

# ---------------------------------------------------------------------------
# Trending pages to scrape (language, period)
# ---------------------------------------------------------------------------
TRENDING_PAGES = [
    ("", "daily"), ("", "weekly"), ("", "monthly"),
    ("python", "daily"), ("python", "monthly"),
    ("typescript", "daily"), ("typescript", "monthly"),
    ("rust", "daily"), ("go", "daily"),
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
REQUEST_DELAY = 1.5

# ---------------------------------------------------------------------------
# Keyword fallback (ONLY used when the LLM proxy is unreachable)
# ---------------------------------------------------------------------------
TIER_1_KEYWORDS = [
    "claude", "anthropic", "claude-code", "mcp", "model context protocol",
    "langchain", "langgraph", "llamaindex", "openai", "chatgpt", "agentic",
    "multi-agent", "ai-agent", "agent framework", "llama", "mistral", "qwen",
    "gemini", "deepseek", "autogen", "crewai", "cursor", "windsurf", "copilot",
    "codex", "ollama", "vllm", "huggingface", "transformers", "openrouter",
    "groq", "mcp-server", "claude-skill", "aider", "cline",
]
# Word-boundary terms (fixes the old bare-substring false positives)
TIER_2_WORD = [
    "ai", "llm", "gpt", "rag", "nlp", "agent", "agents", "embedding",
    "vector", "chatbot", "assistant", "prompt", "lora", "inference",
    "tokenizer", "neural", "diffusion", "skill", "skills",
]
TIER_2_PHRASE = [
    "large language model", "artificial intelligence", "machine learning",
    "retrieval augmented", "vector database", "prompt engineering",
    "fine-tune", "fine-tuning", "function calling", "tool use",
    "deep learning", "natural language", "code generation", "text generation",
]


# ---------------------------------------------------------------------------
# Fetch & Parse
# ---------------------------------------------------------------------------
def fetch_trending(language: str, since: str, session: requests.Session) -> str:
    url = (f"https://github.com/trending/{language}?since={since}"
           if language else f"https://github.com/trending?since={since}")
    logger.info(f"Fetching: {url}")
    for attempt in range(3):
        try:
            resp = session.get(url, headers=REQUEST_HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < 2:
                time.sleep(3)
    logger.error(f"All attempts failed for {url}")
    return ""


def _parse_star_count(text: str) -> int:
    """Parse GitHub star text into an int. Handles '12,345', '1.2k', '12.5k', '1.3m'."""
    if not text:
        return 0
    t = text.strip().lower().replace(",", "")
    m = re.match(r"([\d.]+)\s*([km]?)", t)
    if not m:
        return 0
    try:
        val = float(m.group(1))
    except ValueError:
        return 0
    mult = {"k": 1_000, "m": 1_000_000}.get(m.group(2), 1)
    return int(val * mult)


def parse_trending_html(html: str, language: str, since: str) -> list[dict]:
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    repos = []
    for article in soup.select("article.Box-row"):
        try:
            h2_link = article.select_one("h2 a")
            if not h2_link:
                continue
            repo = {
                "full_name": h2_link.get_text(strip=True).replace(" ", "").replace("\n", "").lstrip("/"),
                "url": f"https://github.com{h2_link.get('href', '')}",
            }
            p_tag = article.select_one("p")
            repo["description"] = p_tag.get_text(strip=True) if p_tag else ""
            lang_span = article.select_one('[itemprop="programmingLanguage"]')
            repo["language"] = lang_span.get_text(strip=True) if lang_span else ""

            star_link = article.select_one('a[href*="/stargazers"]')
            repo["total_stars"] = _parse_star_count(star_link.get_text(strip=True)) if star_link else 0

            gain_span = article.select_one("span.d-inline-block.float-sm-right")
            repo["stars_gained"] = 0
            if gain_span:
                m = re.search(r"([\d.,]+)\s*([km]?)\s+stars?", gain_span.get_text(strip=True).lower())
                if m:
                    repo["stars_gained"] = _parse_star_count(m.group(1) + m.group(2))

            repo["period"] = since
            repo["source_page"] = language or "all"
            repos.append(repo)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to parse article: {e}")
    logger.info(f"  Parsed {len(repos)} repos from {language or 'all'}/{since}")
    return repos


def dedup_repos(all_repos: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    maps = {"daily": {}, "weekly": {}, "monthly": {}}
    for repo in all_repos:
        name = repo["full_name"].lower()
        target = maps.get(repo["period"], maps["daily"])
        if name in target:
            existing = target[name]
            existing["stars_gained"] = max(existing["stars_gained"], repo["stars_gained"])
            existing.setdefault("source_pages", set()).add(repo["source_page"])
        else:
            repo["source_pages"] = {repo["source_page"]}
            target[name] = repo
    out = []
    for period in ("daily", "weekly", "monthly"):
        out.append(sorted(maps[period].values(), key=lambda r: r["stars_gained"], reverse=True))
    return tuple(out)  # type: ignore


# ---------------------------------------------------------------------------
# Profiles + LLM analysis
# ---------------------------------------------------------------------------
def load_profiles() -> dict[str, str]:
    profiles = {}
    for f in sorted(PROFILES_DIR.glob("*.md")):
        if f.name.lower() == "readme.md":
            continue
        profiles[f.stem.lower()] = f.read_text(encoding="utf-8")
    return profiles


def make_client():
    import anthropic
    return anthropic.Anthropic(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def call_llm(client, prompt: str, retries: int = 3, max_tokens: int = 500) -> str:
    """Hardened LLM call — retries + guards empty/blank responses (the same failure
    mode that silently froze the luma pipeline)."""
    last_exc = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=LLM_MODEL, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if not resp.content:
                raise ValueError("empty response.content")
            text = resp.content[0].text
            if not text or not text.strip():
                raise ValueError("blank text")
            return text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise ValueError(f"LLM failed after {retries} attempts: {last_exc}")


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]
    start = raw.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    return obj


def _desc_hash(repo: dict) -> str:
    blob = f"{repo['full_name']}|{repo.get('description', '')}|{repo.get('language', '')}"
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def build_analysis_prompt(profiles: dict[str, str], repo: dict) -> str:
    names = list(profiles.keys())
    profile_sections = "\n\n".join(
        f"PERSON — {n.upper()}:\n{c}" for n, c in profiles.items()
    )
    person_schema = ", ".join(
        f'"{n}": {{"score": <0-100 int>, "reason": "one sentence, why it matters to them"}}'
        for n in names
    )
    cats = ", ".join(CATEGORIES)
    return f"""You are analyzing a trending GitHub repository for {len(names)} people.

{profile_sections}

REPOSITORY:
- Name: {repo['full_name']}
- Description: {repo.get('description') or '(none)'}
- Language: {repo.get('language') or 'unknown'}
- Total stars: {repo.get('total_stars', 0)}
- Stars gained ({repo.get('period', 'daily')}): {repo.get('stars_gained', 0)}

First infer what this repository ACTUALLY is and does (use the name + description; do
not just match keywords). Then score its relevance 0-100 for each person against their
profile. Use the FULL range and non-round numbers. No minimum floor — an off-domain repo
scores 3-10. Reserve 85-100 for repos squarely in a person's core interests.

Pick ONE category from: {cats}

Respond ONLY with valid JSON, no other text:
{{"what_it_is": "one concise sentence — what the repo is/does", "category": "<one of the categories>", {person_schema}}}"""


def analyze_repo(client, profiles: dict[str, str], repo: dict) -> dict | None:
    try:
        raw = call_llm(client, build_analysis_prompt(profiles, repo))
        data = _extract_json(raw)
        # normalize
        out = {
            "what_it_is": str(data.get("what_it_is", "")).strip()[:240],
            "category": data.get("category") if data.get("category") in CATEGORIES else "Other",
            "scores": {},
        }
        for name in profiles:
            s = data.get(name) or {}
            try:
                score = max(0, min(100, int(s.get("score", 0))))
            except (ValueError, TypeError):
                score = 0
            out["scores"][name] = {"score": score, "reason": str(s.get("reason", "")).strip()[:200]}
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  analyze error for {repo['full_name']}: {exc}")
        return None


def analyze_all(unique_repos: list[dict], profiles: dict[str, str], force: bool) -> dict[str, dict]:
    """Return {full_name: analysis}. Cached by (full_name, desc_hash). LLM proxy required;
    caller handles the fallback if this raises on the very first call."""
    cache = {}
    if ANALYSIS_CACHE_PATH.exists() and not force:
        try:
            cache = json.loads(ANALYSIS_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cache = {}

    results: dict[str, dict] = {}
    to_do = []
    for repo in unique_repos:
        h = _desc_hash(repo)
        c = cache.get(repo["full_name"])
        if c and c.get("desc_hash") == h and not force:
            results[repo["full_name"]] = c["analysis"]
        else:
            to_do.append(repo)

    logger.info(f"Analysis: {len(results)} cached, {len(to_do)} to analyze via LLM")
    if to_do:
        client = make_client()
        # Probe once so a dead proxy fails fast (caller falls back to keywords)
        probe = analyze_repo(client, profiles, to_do[0])
        if probe is None:
            raise RuntimeError("LLM analysis probe failed — proxy likely down")
        results[to_do[0]["full_name"]] = probe
        cache[to_do[0]["full_name"]] = {"desc_hash": _desc_hash(to_do[0]), "analysis": probe}

        rest = to_do[1:]
        if rest:
            def work(r):
                return r, analyze_repo(make_client(), profiles, r)
            done = 0
            with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as ex:
                futs = {ex.submit(work, r): r for r in rest}
                for fut in as_completed(futs):
                    r, a = fut.result()
                    done += 1
                    if a:
                        results[r["full_name"]] = a
                        cache[r["full_name"]] = {"desc_hash": _desc_hash(r), "analysis": a}
                    if done % 25 == 0 or done == len(rest):
                        logger.info(f"  analyzed {done}/{len(rest)}")
        ANALYSIS_CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def exec_summary(client, top_repos: list[dict], profiles: dict[str, str]) -> str:
    """One personalized 2-3 sentence synthesis of the day."""
    names = list(profiles.keys())
    lines = []
    for r in top_repos[:12]:
        a = r.get("analysis", {})
        sc = " ".join(f"{n[:1].upper()}={a.get('scores', {}).get(n, {}).get('score', 0)}" for n in names)
        lines.append(f"- {r['full_name']} [{a.get('category','?')}] (+{r['stars_gained']} today, {sc}): {a.get('what_it_is','')}")
    repos_text = "\n".join(lines)
    people = " and ".join(n.capitalize() for n in names)
    try:
        raw = call_llm(client, f"""These are today's top trending AI/dev GitHub repos, with relevance scores for {people}:

{repos_text}

Write a punchy 2-3 sentence briefing ("Today for you: ...") that calls out what's genuinely
new or notable and who each item matters to. Be specific and concrete. Plain text only.""",
                       max_tokens=280)
        return raw.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"exec_summary failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Keyword fallback (proxy down)
# ---------------------------------------------------------------------------
def keyword_relevance(repo: dict) -> tuple[int, str]:
    text = f"{repo['full_name']} {repo.get('description', '')}".lower()
    score = 0
    for kw in TIER_1_KEYWORDS:
        if kw in text:
            score += 10
    if score == 0:
        for w in TIER_2_WORD:
            if re.search(rf"\b{re.escape(w)}\b", text):
                score += 1
        for ph in TIER_2_PHRASE:
            if ph in text:
                score += 1
    # Map keyword score to a pseudo relevance 0-100
    return min(100, score * 8), "keyword match (LLM analysis unavailable)"


# ---------------------------------------------------------------------------
# Novelty tracking
# ---------------------------------------------------------------------------
def load_seen() -> dict:
    if SEEN_PATH.exists():
        try:
            return json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def classify_novelty(daily: list[dict], seen: dict, today: str) -> None:
    """Tag each daily repo with novelty in-place using PRE-update state."""
    for r in daily:
        prior = seen.get(r["full_name"])
        gain = r["stars_gained"]
        if prior is None:
            r["novelty"] = "new"
        else:
            prior_max = prior.get("max_stars_gained", 0)
            if gain >= max(400, int(prior_max * 1.6)) and gain >= 200:
                r["novelty"] = "surging"
            else:
                r["novelty"] = "recurring"


def update_seen(daily: list[dict], seen: dict, today: str) -> None:
    for r in daily:
        name = r["full_name"]
        prior = seen.get(name, {})
        seen[name] = {
            "first_seen": prior.get("first_seen", today),
            "last_seen": today,
            "days_seen": prior.get("days_seen", 0) + 1,
            "max_stars_gained": max(prior.get("max_stars_gained", 0), r["stars_gained"]),
        }
    SEEN_PATH.write_text(json.dumps(seen, indent=2, ensure_ascii=False), encoding="utf-8")


def attach_analysis(repos: list[dict], analyses: dict, profiles: dict) -> None:
    for r in repos:
        a = analyses.get(r["full_name"])
        if not a:
            score, reason = keyword_relevance(r)
            a = {
                "what_it_is": (r.get("description") or "")[:200],
                "category": "Other",
                "scores": {n: {"score": score, "reason": reason} for n in profiles},
            }
        r["analysis"] = a
        r["relevance"] = max((s["score"] for s in a["scores"].values()), default=0)
        r["top_person"] = max(a["scores"], key=lambda n: a["scores"][n]["score"]) if a["scores"] else ""


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
_PERSON_COLORS = [("#7c3aed", "#f3e8ff"), ("#db2777", "#fce7f3"),
                  ("#0284c7", "#e0f2fe"), ("#059669", "#d1fae5")]
_CAT_COLORS = {
    "Agents": ("#fee2e2", "#dc2626"), "Coding Tools": ("#e0e7ff", "#4338ca"),
    "Models": ("#dbeafe", "#2563eb"), "RAG/Search": ("#fef3c7", "#d97706"),
    "MCP": ("#f3e8ff", "#7c3aed"), "Infra": ("#f1f5f9", "#475569"),
    "Apps": ("#d1fae5", "#059669"), "Research": ("#fae8ff", "#a21caf"),
    "Other": ("#f3f4f6", "#6b7280"),
}


def _fmt(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def _person_badges(repo: dict, profiles: dict) -> str:
    out = ""
    for i, name in enumerate(profiles):
        sc = repo["analysis"]["scores"].get(name, {}).get("score", 0)
        fg, bg = _PERSON_COLORS[i % len(_PERSON_COLORS)]
        strong = sc >= RELEVANCE_THRESHOLD
        style = (f"background:{bg};color:{fg};" if strong
                 else "background:#f3f4f6;color:#9ca3af;")
        out += (f'<span title="{name.capitalize()}: {sc}" style="{style}'
                f'padding:2px 7px;border-radius:8px;font-size:10px;font-weight:800;'
                f'margin-right:3px;">{name[:1].upper()} {sc}</span>')
    return out


def _cat_chip(cat: str) -> str:
    bg, fg = _CAT_COLORS.get(cat, _CAT_COLORS["Other"])
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:8px;'
            f'font-size:10px;font-weight:700;">{cat}</span>')


def _repo_card(repo: dict, profiles: dict, i: int, compact: bool = False) -> str:
    a = repo["analysis"]
    name = repo["full_name"]
    bg = "#ffffff" if i % 2 == 0 else "#f9fafb"
    gain = repo["stars_gained"]
    gain_bg, gain_fg = (("#fef3c7", "#92400e") if gain >= 1000
                        else ("#d1fae5", "#065f46") if gain >= 300 else ("#eff6ff", "#1e40af"))
    novelty = repo.get("novelty", "")
    nb = ""
    if novelty == "new":
        nb = '<span style="background:#7c3aed;color:#fff;padding:2px 7px;border-radius:8px;font-size:10px;font-weight:800;margin-right:4px;">&#128308; NEW</span>'
    elif novelty == "surging":
        nb = '<span style="background:#ea580c;color:#fff;padding:2px 7px;border-radius:8px;font-size:10px;font-weight:800;margin-right:4px;">&#128293; SURGING</span>'

    if compact:
        return f"""<tr style="background:{bg};border-bottom:1px solid #f3f4f6;">
            <td style="padding:8px 12px;">
              <a href="{repo['url']}" style="color:#111827;font-weight:600;font-size:13px;text-decoration:none;">{name}</a>
              <span style="color:#9ca3af;font-size:11px;margin-left:6px;">{a.get('what_it_is','')[:70]}</span>
            </td>
            <td style="padding:8px 10px;text-align:right;white-space:nowrap;">{_person_badges(repo, profiles)}
              <span style="color:#6b7280;font-size:11px;margin-left:5px;">&#11088;{_fmt(repo['total_stars'])}</span>
              <span style="background:{gain_bg};color:{gain_fg};padding:3px 9px;border-radius:10px;font-weight:800;font-size:12px;margin-left:5px;">+{gain:,}</span></td>
        </tr>"""

    top = repo.get("top_person", "")
    reason = a["scores"].get(top, {}).get("reason", "") if top else ""
    reason_html = (f'<div style="color:#4b5563;font-size:12px;margin-top:5px;line-height:1.45;">'
                   f'<b style="color:#7c3aed;">&rarr; {top.capitalize()}:</b> {reason}</div>') if reason else ""
    return f"""<tr style="background:{bg};border-bottom:1px solid #f3f4f6;">
        <td style="padding:13px 14px;">
          <div style="margin-bottom:4px;">{nb}{_cat_chip(a.get('category','Other'))}</div>
          <a href="{repo['url']}" style="color:#111827;font-weight:700;font-size:14px;text-decoration:none;">{name}</a>
          <div style="color:#374151;font-size:12.5px;margin-top:3px;line-height:1.45;">{a.get('what_it_is','')}</div>
          {reason_html}
          <div style="margin-top:6px;">{_person_badges(repo, profiles)}</div>
        </td>
        <td style="padding:13px 10px;text-align:center;white-space:nowrap;vertical-align:top;">
          <span style="background:{gain_bg};color:{gain_fg};padding:4px 12px;border-radius:12px;font-weight:800;font-size:13px;">+{gain:,}</span>
          <div style="font-size:11px;color:#9ca3af;margin-top:4px;">&#11088; {_fmt(repo['total_stars'])} total &middot; {repo.get('language') or '—'}</div>
        </td>
    </tr>"""


def _table(rows: str) -> str:
    return f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;"><tbody>{rows}</tbody></table>'


def _section(title: str, repos: list[dict], profiles: dict, compact: bool = False) -> str:
    if not repos:
        return ""
    rows = "".join(_repo_card(r, profiles, i, compact) for i, r in enumerate(repos))
    return f"""<div style="background:#ffffff;padding:18px 28px 6px;border-left:1px solid #e5e7eb;border-right:1px solid #e5e7eb;border-top:2px solid #f3f4f6;">
        <h2 style="margin:0 0 10px;font-size:16px;font-weight:800;color:#111827;">{title}</h2>
        {_table(rows)}
    </div>"""


def generate_html_report(daily_rel, weekly_rel, monthly_rel, daily_all,
                         profiles, report_date, summary_text, recent_links) -> str:
    new = [r for r in daily_rel if r.get("novelty") == "new"]
    surging = [r for r in daily_rel if r.get("novelty") == "surging"]
    recurring = [r for r in daily_rel if r.get("novelty") == "recurring"]

    # By-category grouping for recurring relevant
    cat_html = ""
    if recurring:
        groups: dict[str, list] = {}
        for r in recurring:
            groups.setdefault(r["analysis"].get("category", "Other"), []).append(r)
        blocks = ""
        for cat in CATEGORIES:
            if cat in groups:
                rows = "".join(_repo_card(r, profiles, i, compact=True) for i, r in enumerate(groups[cat]))
                blocks += f'<div style="margin-bottom:6px;"><div style="font-size:12px;font-weight:800;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;margin:10px 0 4px;">{cat}</div>{_table(rows)}</div>'
        cat_html = f"""<div style="background:#ffffff;padding:18px 28px 6px;border-left:1px solid #e5e7eb;border-right:1px solid #e5e7eb;border-top:2px solid #f3f4f6;">
            <h2 style="margin:0 0 6px;font-size:16px;font-weight:800;color:#111827;">&#128193; Still Relevant &mdash; by Category</h2>
            {blocks}</div>"""

    # Weekly / monthly context (compact, relevant only)
    wm_rows = "".join(_repo_card(r, profiles, i, compact=True) for i, r in enumerate((weekly_rel + monthly_rel)[:12]))
    wm_html = f"""<div style="background:#f8fafc;padding:16px 28px;border-left:1px solid #e5e7eb;border-right:1px solid #e5e7eb;border-top:2px solid #f3f4f6;">
        <h2 style="margin:0 0 8px;font-size:14px;font-weight:700;color:#9ca3af;">&#128200; This Week / Month (context)</h2>
        {_table(wm_rows)}</div>""" if wm_rows else ""

    # Also trending (non-relevant top gainers)
    rel_names = {r["full_name"].lower() for r in daily_rel}
    others = [r for r in daily_all if r["full_name"].lower() not in rel_names][:6]
    other_rows = "".join(
        f'<tr style="background:{"#ffffff" if i%2==0 else "#f9fafb"};"><td style="padding:7px 12px;"><a href="{r["url"]}" style="color:#6b7280;font-size:12.5px;text-decoration:none;">{r["full_name"]}</a></td><td style="padding:7px 10px;text-align:right;font-size:12px;color:#9ca3af;white-space:nowrap;">&#11088;{_fmt(r["total_stars"])} <span style="color:#6b7280;margin-left:4px;">+{r["stars_gained"]:,}</span></td></tr>'
        for i, r in enumerate(others))
    other_html = f"""<div style="background:#f8fafc;padding:14px 28px;border-left:1px solid #e5e7eb;border-right:1px solid #e5e7eb;">
        <h2 style="margin:0 0 6px;font-size:13px;font-weight:700;color:#9ca3af;">Also Trending (not matched to your profiles)</h2>
        {_table(other_rows)}</div>""" if other_rows else ""

    summary_html = ""
    if summary_text:
        summary_html = f"""<div style="background:#ffffff;padding:18px 28px;border-left:1px solid #e5e7eb;border-right:1px solid #e5e7eb;">
            <div style="background:linear-gradient(135deg,#faf5ff,#eff6ff);border:1px solid #e9d5ff;border-radius:12px;padding:16px 18px;">
              <div style="font-size:12px;font-weight:800;color:#7c3aed;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">&#10024; Today's Briefing</div>
              <div style="font-size:13.5px;color:#374151;line-height:1.55;">{summary_text}</div>
            </div></div>"""

    top_gainer = max(daily_rel, key=lambda r: r["stars_gained"], default=None)
    top_name = top_gainer["full_name"].split("/")[-1] if top_gainer else "—"
    recent_html = ""
    if recent_links:
        recent_html = " &middot; ".join(recent_links)

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;">
<div style="max-width:760px;margin:0 auto;padding:24px 14px;">
  <div style="background:linear-gradient(135deg,#7c3aed,#a855f7,#c084fc);border-radius:16px 16px 0 0;padding:30px 28px 24px;text-align:center;">
    <div style="font-size:30px;">&#128293;</div>
    <h1 style="margin:6px 0 0;color:#fff;font-size:22px;font-weight:800;">GitHub Trending &mdash; for Sumeet &amp; Ayushi</h1>
    <p style="margin:6px 0 0;color:rgba(255,255,255,.85);font-size:13px;">{report_date}</p>
    <a href="{PAGES_URL}" style="display:inline-block;margin-top:14px;background:#fff;color:#7c3aed;padding:9px 20px;border-radius:20px;font-weight:800;font-size:13px;text-decoration:none;">&#128241; Open on your phone</a>
  </div>
  <div style="background:#ffffff;padding:20px 28px 8px;border-left:1px solid #e5e7eb;border-right:1px solid #e5e7eb;">
    <table width="100%"><tr>
      <td width="33%" style="text-align:center;"><div style="background:#faf5ff;border-radius:12px;padding:14px 6px;"><div style="font-size:26px;font-weight:800;color:#7c3aed;">{len(daily_rel)}</div><div style="font-size:10px;color:#7c3aed;text-transform:uppercase;">Relevant Today</div></div></td>
      <td width="33%" style="text-align:center;"><div style="background:#eff6ff;border-radius:12px;padding:14px 6px;"><div style="font-size:26px;font-weight:800;color:#2563eb;">{len(new)}</div><div style="font-size:10px;color:#2563eb;text-transform:uppercase;">New Today</div></div></td>
      <td width="33%" style="text-align:center;"><div style="background:#fef3c7;border-radius:12px;padding:14px 6px;"><div style="font-size:15px;font-weight:800;color:#92400e;line-height:1.3;">{top_name}</div><div style="font-size:10px;color:#92400e;text-transform:uppercase;">Top Gainer</div></div></td>
    </tr></table>
  </div>
  {summary_html}
  {_section("&#128308; New Today", new, profiles)}
  {_section("&#128293; Surging", surging, profiles)}
  {cat_html}
  {wm_html}
  {other_html}
  <div style="background:#faf5ff;border-radius:0 0 16px 16px;padding:18px 28px;border:1px solid #e5e7eb;border-top:none;text-align:center;">
    <p style="margin:0;font-size:12px;color:#6b7280;">&#129302; Analyzed & scored against your profiles &middot; <a href="{PAGES_URL}" style="color:#7c3aed;">web version</a></p>
    <p style="margin:6px 0 0;font-size:11px;color:#9ca3af;">Recent: {recent_html}</p>
  </div>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Publish to GitHub Pages
# ---------------------------------------------------------------------------
def publish_to_pages(html: str, date_str: str) -> bool:
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    (ARCHIVE_DIR / f"{date_str}.html").write_text(html, encoding="utf-8")

    def git(*a):
        return subprocess.run(["git"] + list(a), cwd=str(SCRIPT_DIR),
                              capture_output=True, text=True, timeout=60)

    def gh(*a):
        return subprocess.run(["gh"] + list(a), cwd=str(SCRIPT_DIR),
                              capture_output=True, text=True, timeout=60)

    if not git("status", "--porcelain", "docs").stdout.strip():
        logger.info("[publish] docs unchanged — nothing to push")
        return True
    git("add", "docs")
    c = git("commit", "-m", f"Digest {date_str}")
    if c.returncode != 0:
        logger.error(f"[publish] commit failed: {c.stderr.strip()}")
        return False
    sw = gh("auth", "switch", "--user", PAGES_ACCOUNT)
    if sw.returncode != 0:
        logger.warning(f"[publish] gh switch failed: {sw.stderr.strip()}")
    try:
        p = git("push", "origin", PAGES_BRANCH)
    finally:
        gh("auth", "switch", "--user", PAGES_RESTORE_ACCOUNT)
    if p.returncode != 0:
        logger.error(f"[publish] push failed: {p.stderr.strip()}")
        return False
    logger.info(f"[publish] pushed to {PAGES_REPO} — {PAGES_URL}")
    return True


def recent_digest_links() -> list[str]:
    files = sorted(ARCHIVE_DIR.glob("*.html"), reverse=True)[:7]
    out = []
    for f in files:
        d = f.stem
        out.append(f'<a href="{PAGES_URL}archive/{d}.html" style="color:#9ca3af;">{d[5:]}</a>')
    return out


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
def send_report_email(html: str, subject: str) -> bool:
    if not EMAIL_TO:
        logger.error("No EMAIL_TO configured (set GHTREND_EMAIL_TO or config.local.json)")
        return False
    try:
        sys.path.insert(0, str(BROWSER_AGENT))
        from linkedin_apply.email_report import send_email
        result = send_email(subject=subject, html_body=html, to=EMAIL_TO)
        logger.info(f"Email sent to {EMAIL_TO} — ID: {result.get('id','N/A') if result else 'N/A'}")
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to send email: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GitHub Trending — personalized digest")
    parser.add_argument("--dry-run", action="store_true", help="Save only; no email/publish")
    parser.add_argument("--no-email", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore analysis cache")
    args = parser.parse_args()

    report_date = datetime.now().strftime("%B %d, %Y")
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== GitHub Trending — {report_date} ===")

    profiles = load_profiles()
    if not profiles:
        logger.error("No profiles found in profiles/*.md")
        sys.exit(1)
    logger.info(f"Profiles: {', '.join(profiles)}")

    # 1. Fetch
    session = requests.Session()
    all_repos, pages_ok = [], 0
    for lang, since in TRENDING_PAGES:
        html = fetch_trending(lang, since, session)
        if html:
            pages_ok += 1
        all_repos.extend(parse_trending_html(html, lang, since))
        time.sleep(REQUEST_DELAY)
    logger.info(f"Fetched {len(all_repos)} raw repos from {pages_ok}/{len(TRENDING_PAGES)} pages")
    if pages_ok < 6:
        logger.warning(f"Only {pages_ok}/{len(TRENDING_PAGES)} pages fetched OK — partial data")
    if not all_repos:
        logger.error("No repos fetched.")
        sys.exit(1)

    # 2. Dedup per period
    daily_all, weekly_all, monthly_all = dedup_repos(all_repos)
    logger.info(f"Dedup: {len(daily_all)} daily, {len(weekly_all)} weekly, {len(monthly_all)} monthly")

    # 3. LLM analysis on the global unique set (cached), with keyword fallback
    unique = {}
    for r in daily_all + weekly_all + monthly_all:
        unique.setdefault(r["full_name"], r)
    unique_list = list(unique.values())

    analyses = {}
    try:
        analyses = analyze_all(unique_list, profiles, args.force)
        logger.info(f"LLM analysis complete for {len(analyses)} repos")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"LLM analysis unavailable ({exc}) — falling back to keyword scoring")

    for lst in (daily_all, weekly_all, monthly_all):
        attach_analysis(lst, analyses, profiles)

    # 4. Relevance filter
    def relevant(lst):
        return sorted([r for r in lst if r["relevance"] >= RELEVANCE_THRESHOLD],
                      key=lambda r: (r["relevance"], r["stars_gained"]), reverse=True)
    daily_rel, weekly_rel, monthly_rel = relevant(daily_all), relevant(weekly_all), relevant(monthly_all)
    logger.info(f"Relevant (>= {RELEVANCE_THRESHOLD}): {len(daily_rel)} daily, "
                f"{len(weekly_rel)} weekly, {len(monthly_rel)} monthly")

    # 5. Novelty (classify with pre-update state, then persist)
    seen = load_seen()
    classify_novelty(daily_rel, seen, today)
    n_new = sum(1 for r in daily_rel if r["novelty"] == "new")
    n_surge = sum(1 for r in daily_rel if r["novelty"] == "surging")
    logger.info(f"Novelty: {n_new} new, {n_surge} surging, {len(daily_rel)-n_new-n_surge} recurring")
    update_seen(daily_all, seen, today)

    # 6. Exec summary (best-effort)
    summary_text = ""
    if analyses and daily_rel:
        try:
            summary_text = exec_summary(make_client(), daily_rel, profiles)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"exec summary skipped: {exc}")

    # 7. Render
    html = generate_html_report(daily_rel, weekly_rel, monthly_rel, daily_all,
                                profiles, report_date, summary_text, recent_digest_links())
    report_path = OUTPUT_DIR / f"report_{today}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"Report saved: {report_path}")

    for r in daily_rel[:15]:
        logger.info(f"  [{r.get('novelty','')[:4].upper():4}] +{r['stars_gained']:,} {r['full_name']} "
                    f"[{r['analysis'].get('category')}] rel={r['relevance']}")

    if args.dry_run:
        logger.info("DRY RUN — no email, no publish")
        sys.exit(0)

    # 8. Publish (never fatal)
    if not args.no_publish:
        try:
            publish_to_pages(html, today)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[publish] error (non-fatal): {exc}")

    # 9. Email
    if args.no_email:
        sys.exit(0)
    subject = f"GitHub Trending — {report_date} ({len(daily_rel)} relevant, {n_new} new)"
    ok = send_report_email(html, subject)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
