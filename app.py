"""
ARORA RAIN - AI Learning Companion
===================================
Run with:  streamlit run app.py

The file is organised into clearly separated sections (think of each one as a
module). Every section has a comment explaining which library it uses and why.

  1. CONFIG & STYLING      -> Streamlit (UI, CSS gradients, animations)
  2. STORAGE               -> JSON file + Pandas (user progress)
  3. AI ENGINE             -> Requests (Claude API). Offline fallback below.
  4. OFFLINE ENGINE        -> Requests + BeautifulSoup4 (Wikipedia) +
                              Scikit-learn (TF-IDF) + NumPy
  5. DOCUMENT EXPORT       -> python-docx (Word) + FPDF2 (PDF)
  6. CHARTS                -> Plotly, Matplotlib, Seaborn
  7. PAGES                 -> Streamlit pages selected from the sidebar
"""

import io
import json
import os
import random
import re
import uuid
from collections import Counter
from datetime import date, datetime, timedelta
from urllib.parse import quote_plus

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from docx import Document
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 1. CONFIG & STYLING  (Streamlit)
# =============================================================================
APP_NAME = "Arora Rain"
DATA_FILE = "arora_data.json"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"
WIKI_API = "https://en.wikipedia.org/w/api.php"
HTTP_HEADERS = {"User-Agent": "AroraRain/1.0 (educational learning app)"}

st.set_page_config(page_title=f"{APP_NAME} - AI Learning Companion",
                   page_icon="🌧️", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
@keyframes fadeUp { from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:translateY(0);} }
@keyframes shimmer { 0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;} }
.main .block-container { animation: fadeUp .6s ease; padding-top: 1.5rem; max-width: 1200px; }
.hero {
  background: linear-gradient(120deg,#4f46e5,#7c3aed,#06b6d4,#4f46e5);
  background-size: 300% 300%; animation: shimmer 10s ease infinite;
  padding: 28px 32px; border-radius: 20px; color: white; margin-bottom: 22px;
  box-shadow: 0 10px 30px rgba(79,70,229,.35);
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
.hero p { margin: 6px 0 0; opacity: .92; font-size: 1.02rem; }
.card {
  background: rgba(255,255,255,.04); border: 1px solid rgba(124,58,237,.25);
  border-radius: 16px; padding: 18px 20px; margin-bottom: 14px;
  transition: transform .25s ease, box-shadow .25s ease; animation: fadeUp .6s ease;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(124,58,237,.25); }
.card h4 { margin: 0 0 6px; }
.pill { display:inline-block; padding:3px 12px; border-radius:999px; font-size:.78rem;
        background:linear-gradient(90deg,#4f46e5,#06b6d4); color:white; margin-right:6px; }
div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
  background: linear-gradient(90deg,#4f46e5,#7c3aed); color: white; border: 0;
  border-radius: 12px; padding: .55rem 1.3rem; font-weight: 600; transition: all .25s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-2px); box-shadow: 0 8px 18px rgba(79,70,229,.45); color: white;
}
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#1e1b4b 0%,#312e81 55%,#0e7490 100%); }
section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
[data-testid="stMetric"] { background: rgba(124,58,237,.08); border-radius: 14px; padding: 12px 16px;
                           border: 1px solid rgba(124,58,237,.2); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def card(title, body_html):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body_html}</div>',
                unsafe_allow_html=True)


# =============================================================================
# 2. STORAGE  (JSON file + Pandas)
# =============================================================================
def default_data():
    return {"completed_topics": [], "exams": [], "activity": [], "roadmap": None}


def load_data():
    """Load saved progress. A corrupted or missing file yields fresh data."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in default_data().items():
                data.setdefault(k, v)
            return data
        except (json.JSONDecodeError, OSError):
            return default_data()
    return default_data()


def save_data(data):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        st.warning(f"Could not save progress: {e}")


def log_activity(kind, detail=""):
    data = load_data()
    data["activity"].append({"date": date.today().isoformat(), "type": kind, "detail": detail})
    save_data(data)


def mark_completed(topic, source="manual"):
    data = load_data()
    if not any(t["topic"].lower() == topic.lower() for t in data["completed_topics"]):
        data["completed_topics"].append(
            {"topic": topic, "source": source, "date": date.today().isoformat()})
        save_data(data)


def unmark_completed(topic):
    data = load_data()
    data["completed_topics"] = [t for t in data["completed_topics"]
                                if t["topic"].lower() != topic.lower()]
    save_data(data)


def record_exam(topic, difficulty, score, total):
    data = load_data()
    data["exams"].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "topic": topic,
        "difficulty": difficulty, "score": score, "total": total,
        "percent": round(100 * score / total, 1) if total else 0})
    save_data(data)


def exams_dataframe():
    data = load_data()
    return pd.DataFrame(data["exams"]) if data["exams"] else pd.DataFrame(
        columns=["date", "topic", "difficulty", "score", "total", "percent"])


def current_streak():
    dates = sorted({a["date"] for a in load_data()["activity"]}, reverse=True)
    if not dates:
        return 0
    streak, day = 0, date.today()
    if dates[0] != day.isoformat():       # allow streak to continue from yesterday
        day -= timedelta(days=1)
    for d in dates:
        if d == day.isoformat():
            streak += 1
            day -= timedelta(days=1)
        elif d < day.isoformat():
            break
    return streak


# =============================================================================
# 3. AI ENGINE  (Requests -> Claude API)
# =============================================================================
class AIError(Exception):
    pass


def get_api_key():
    key = st.session_state.get("api_key", "").strip()
    if key:
        return key
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip()
    except Exception:
        return ""


def ai_enabled():
    return bool(get_api_key())


def call_llm(prompt, system="You are Arora Rain, a friendly expert teacher.",
             max_tokens=4000, temperature=0.4):
    """Send one prompt to the Claude Messages API using the Requests library."""
    key = get_api_key()
    if not key:
        raise AIError("No API key configured.")
    payload = {"model": st.session_state.get("model", DEFAULT_MODEL),
               "max_tokens": max_tokens, "temperature": temperature,
               "system": system, "messages": [{"role": "user", "content": prompt}]}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    try:
        r = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=120)
    except requests.RequestException as e:
        raise AIError(f"Network error: {e}")
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", r.text)
        except ValueError:
            msg = r.text
        raise AIError(f"API error {r.status_code}: {msg}")
    blocks = r.json().get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def extract_json(text):
    """Pull the first JSON object/array out of an LLM reply."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise AIError("The AI reply did not contain JSON.")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise AIError(f"Could not parse AI JSON: {e}")


def with_fallback(ai_func, offline_func, what):
    """Use the AI when a key exists; fall back to the offline engine on failure."""
    if ai_enabled():
        try:
            return ai_func(), "ai"
        except AIError as e:
            st.warning(f"AI request failed ({e}). Using the offline engine instead.")
    else:
        st.info(f"No API key set - using the offline engine for {what}. "
                "Add a Claude API key in the sidebar for richer results.")
    return offline_func(), "offline"


# ---- AI prompts -------------------------------------------------------------
def ai_explain(topic, level):
    prompt = (f"Explain the topic '{topic}' for a {level} learner in Markdown with these sections: "
              "## Simple Explanation, ## Key Concepts (bullets), ## Examples (2-3 concrete ones), "
              "## Common Mistakes, ## Quick Recap. Keep it clear and friendly.")
    return call_llm(prompt, max_tokens=2500)


def ai_summary(text):
    prompt = ("Summarise the lecture text below. Reply ONLY with JSON: "
              '{"short": "3-4 sentence summary", "detailed": "multi-paragraph detailed summary", '
              '"key_points": ["important point", ...]}\n\nTEXT:\n' + text[:15000])
    d = extract_json(call_llm(prompt, max_tokens=3000))
    return {"short": d.get("short", ""), "detailed": d.get("detailed", ""),
            "key_points": d.get("key_points", [])}


def ai_notes(text, title):
    prompt = ("Turn the lecture content into organised study notes. Reply ONLY with JSON: "
              '{"title": str, "sections": [{"heading": str, "points": [str]}], '
              '"definitions": [{"term": str, "meaning": str}], "examples": [str], '
              '"key_points": [str]}\n\nTITLE: ' + title + "\nCONTENT:\n" + text[:15000])
    d = extract_json(call_llm(prompt, max_tokens=3500))
    d.setdefault("title", title)
    for k in ("sections", "definitions", "examples", "key_points"):
        d.setdefault(k, [])
    return d


def ai_exam(topic, difficulty, n):
    questions, seen = [], []
    while len(questions) < n:
        batch = min(15, n - len(questions))
        avoid = ("\nDo NOT repeat these questions: " + " | ".join(seen[-40:])) if seen else ""
        prompt = (f"Create {batch} multiple-choice questions about '{topic}'. Difficulty: {difficulty}"
                  " (if Mixed, vary easy/medium/hard). Reply ONLY with a JSON array of objects: "
                  '{"question": str, "options": [4 strings], "answer": index 0-3 of the correct option, '
                  '"explanation": str, "difficulty": "Easy|Medium|Hard"}. '
                  "Exactly one option must be correct." + avoid)
        added = 0
        for q in extract_json(call_llm(prompt, max_tokens=6000, temperature=0.7)):
            try:
                opts = [str(o) for o in q["options"]][:4]
                ans = int(q["answer"])
                if len(opts) == 4 and 0 <= ans < 4 and q["question"] not in seen:
                    questions.append({"question": str(q["question"]), "options": opts,
                                      "answer": ans, "explanation": str(q.get("explanation", "")),
                                      "difficulty": str(q.get("difficulty", difficulty))})
                    seen.append(str(q["question"]))
                    added += 1
            except (KeyError, ValueError, TypeError):
                continue
        if added == 0:
            raise AIError("The AI returned no valid questions.")
    return questions[:n]


def ai_roadmap(field, level, hours):
    prompt = (f"Create a personalised learning roadmap. Field: {field}. Level: {level}. "
              f"Available time: {hours} hours per week. Reply ONLY with JSON: "
              '{"title": str, "summary": str, "total_weeks": int, "phases": [{"name": str, '
              '"weeks": int, "topics": [str], "milestone": str, '
              '"resources": [{"title": str, "type": "Course|Book|Video|Practice|Docs"}]}]}. '
              "Use 4-6 phases and 3-6 topics per phase. Resource titles should be real, well-known resources.")
    d = extract_json(call_llm(prompt, max_tokens=3500))
    if not isinstance(d, dict) or not d.get("phases"):
        raise AIError("Roadmap JSON was incomplete.")
    d.setdefault("title", f"{field} Roadmap")
    d.setdefault("summary", "")
    d.setdefault("total_weeks", sum(int(p.get("weeks", 1)) for p in d["phases"]))
    return d


# =============================================================================
# 4. OFFLINE ENGINE  (Requests + BeautifulSoup4 + Scikit-learn + NumPy)
# =============================================================================
STOPWORDS = set("""a about above after again all also an and any are as at be because been before being
between both but by can could did do does doing down during each few for from further had has have having he
her here him his how i if in into is it its just may me more most my no nor not of off on once only or other
our out over own same she should so some such than that the their them then there these they this those
through to too under until up very was we were what when where which while who why will with would you your
however although among many much used use using called known one two three also often""".split())

SKIP_HEADINGS = {"references", "external links", "see also", "notes", "further reading",
                 "bibliography", "sources", "footnotes", "citations"}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_wikipedia(topic):
    """Find a Wikipedia article and parse it with BeautifulSoup. Returns a dict."""
    try:
        s = requests.get(WIKI_API, headers=HTTP_HEADERS, timeout=12, params={
            "action": "query", "list": "search", "srsearch": topic,
            "srlimit": 1, "format": "json"})
        s.raise_for_status()
        hits = s.json().get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
        p = requests.get(WIKI_API, headers=HTTP_HEADERS, timeout=15, params={
            "action": "parse", "page": title, "prop": "text", "format": "json",
            "redirects": 1, "disableeditsection": 1})
        p.raise_for_status()
        html = p.json()["parse"]["text"]["*"]
    except (requests.RequestException, KeyError, ValueError):
        return None

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select("sup, table, style, .reference, .mw-editsection, .navbox, .hatnote"):
        tag.decompose()
    sections, heading, buf = [], "Introduction", []
    for el in soup.find_all(["h2", "p"]):
        if el.name == "h2":
            if buf:
                sections.append({"heading": heading, "text": " ".join(buf)})
            heading, buf = el.get_text(" ", strip=True), []
        else:
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            if len(t) > 40:
                buf.append(t)
    if buf:
        sections.append({"heading": heading, "text": " ".join(buf)})
    sections = [s for s in sections if s["heading"].lower() not in SKIP_HEADINGS]
    if not sections:
        return None
    return {"title": title, "url": f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
            "sections": sections}


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", re.sub(r"\s+", " ", text).strip())
    return [p.strip() for p in parts if len(p.strip()) > 25]


def rank_sentences(sentences):
    """Score sentences with TF-IDF (Scikit-learn) + NumPy. Returns scores array."""
    if len(sentences) < 2:
        return np.ones(len(sentences))
    try:
        vec = TfidfVectorizer(stop_words="english")
        m = vec.fit_transform(sentences)
    except ValueError:
        return np.ones(len(sentences))
    scores = np.asarray(m.sum(axis=1)).ravel()
    lengths = np.array([len(s.split()) for s in sentences], dtype=float)
    return scores / np.sqrt(np.maximum(lengths, 1))       # mild length normalisation


def top_sentences(sentences, k):
    if not sentences:
        return []
    k = max(1, min(k, len(sentences)))
    idx = np.sort(np.argsort(rank_sentences(sentences))[::-1][:k])   # keep original order
    return [sentences[i] for i in idx]


def top_keywords(text, k=10):
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=500)
        m = vec.fit_transform(split_sentences(text) or [text])
        scores = np.asarray(m.sum(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        return [t for t in terms[np.argsort(scores)[::-1][:k]]]
    except ValueError:
        return []


def offline_summary(text):
    sents = split_sentences(text)
    if len(sents) < 3:
        raise ValueError("Please provide a longer text (at least a few sentences).")
    n_detail = int(np.clip(round(len(sents) * 0.25), 6, 15))
    detail = top_sentences(sents, n_detail)
    paragraphs = [" ".join(detail[i:i + 3]) for i in range(0, len(detail), 3)]
    return {"short": " ".join(top_sentences(sents, 3)),
            "detailed": "\n\n".join(paragraphs),
            "key_points": top_sentences(sents, 6)}


def offline_notes(text, title, sections=None):
    if sections is None:
        sections = [{"heading": "Main Content", "text": text}]
    out = {"title": title, "sections": [], "definitions": [], "examples": [], "key_points": []}
    all_sents = []
    for sec in sections[:8]:
        sents = split_sentences(sec["text"])
        all_sents += sents
        pts = top_sentences(sents, 3)
        if pts:
            out["sections"].append({"heading": sec["heading"], "points": pts})
    if sections and len(sections) > 0:
        first = split_sentences(sections[0]["text"])
        if first:
            out["definitions"].append({"term": title, "meaning": first[0]})
    for kw in top_keywords(text, 12):
        for s in all_sents:
            if re.search(rf"\b{re.escape(kw)}\b", s, re.I) and re.search(r"\b(is|are|refers to|means)\b", s):
                if not any(d["term"] == kw for d in out["definitions"]):
                    out["definitions"].append({"term": kw, "meaning": s})
                break
        if len(out["definitions"]) >= 5:
            break
    out["examples"] = [s for s in all_sents if re.search(r"for example|such as|e\.g\.|for instance", s, re.I)][:4]
    out["key_points"] = top_sentences(all_sents, 6)
    if not out["sections"]:
        raise ValueError("Not enough text to build notes.")
    return out


def offline_explanation(topic):
    page = fetch_wikipedia(topic)
    if not page:
        raise ValueError("No offline source found for that topic. Try a different spelling, "
                         "or add a Claude API key in the sidebar.")
    full = " ".join(s["text"] for s in page["sections"])
    intro = split_sentences(page["sections"][0]["text"])
    md = [f"## Simple Explanation\n{' '.join(intro[:3])}\n", "## Key Concepts"]
    md += [f"- **{k.title()}**" for k in top_keywords(full, 8)]
    md.append("\n## Main Ideas")
    for sec in page["sections"][1:6]:
        pts = top_sentences(split_sentences(sec["text"]), 2)
        if pts:
            md.append(f"**{sec['heading']}** - {' '.join(pts)}\n")
    ex = [s for s in split_sentences(full) if re.search(r"for example|such as|e\.g\.|for instance", s, re.I)][:3]
    md.append("## Examples")
    md += [f"- {e}" for e in ex] or ["- (Examples need the AI engine - add a Claude API key in the sidebar.)"]
    md.append(f"\n*Source: [Wikipedia - {page['title']}]({page['url']})*")
    return "\n".join(md), full


def offline_exam(topic, difficulty, n):
    page = fetch_wikipedia(topic)
    if not page:
        raise ValueError("Could not fetch source material for that topic offline.")
    text = " ".join(s["text"] for s in page["sections"])
    sents = [s for s in split_sentences(text) if 50 <= len(s) <= 230]
    words = [w for w in re.findall(r"[A-Za-z]{6,}", text) if w.lower() not in STOPWORDS]
    freq = Counter(w.lower() for w in words)
    pool = list(freq.keys())
    if len(sents) < 4 or len(pool) < 8:
        raise ValueError("Not enough source text for an offline exam.")
    rng = random.Random(hash((topic, difficulty, n)) & 0xFFFF)
    rng.shuffle(sents)
    questions = []
    for s in sents:
        diff = difficulty if difficulty != "Mixed" else rng.choice(["Easy", "Medium", "Hard"])
        cands = [w for w in re.findall(r"[A-Za-z]{6,}", s) if w.lower() in freq]
        if not cands:
            continue
        cands.sort(key=lambda w: -freq[w.lower()])
        target = cands[0] if diff == "Easy" else cands[min(len(cands) - 1, len(cands) // 2)] \
            if diff == "Medium" else cands[-1]
        tl = target.lower()
        if diff == "Hard":                                     # look-alike distractors
            near = [w for w in pool if w != tl and abs(len(w) - len(tl)) <= 2]
            others = rng.sample(near, 3) if len(near) >= 3 else rng.sample([w for w in pool if w != tl], 3)
        else:
            others = rng.sample([w for w in pool if w != tl], 3)
        opts = [tl] + others
        rng.shuffle(opts)
        blank = re.sub(rf"\b{re.escape(target)}\b", "_______", s, count=1)
        questions.append({"question": f"Fill in the blank: {blank}",
                          "options": [o.capitalize() if target[0].isupper() else o for o in opts],
                          "answer": opts.index(tl),
                          "explanation": f"The original sentence reads: \"{s}\"",
                          "difficulty": diff})
        if len(questions) >= n:
            break
    if not questions:
        raise ValueError("Could not build questions from the source text.")
    return questions


def resource_link(title, rtype, field):
    q = quote_plus(f"{title} {field}")
    if rtype.lower() == "video":
        return f"https://www.youtube.com/results?search_query={q}"
    return f"https://www.google.com/search?q={q}"


def offline_roadmap(field, level, hours):
    base_hours = {"Beginner": 120, "Intermediate": 90, "Advanced": 70}[level]
    total_weeks = int(np.clip(np.ceil(base_hours / max(hours, 1)), 4, 52))
    names = {"Beginner": [("Foundations", ["Core terminology", "Basic tools & setup", "Fundamental principles"]),
                          ("Core Concepts", ["Key theories", "Guided exercises", "Common patterns"]),
                          ("Hands-on Practice", ["Small exercises", "Mini project", "Debugging & feedback"]),
                          ("Review & Next Steps", ["Revision", "Self-assessment quiz", "Plan next level"])],
             "Intermediate": [("Strengthen Basics", ["Gap analysis", "Core concepts revisited"]),
                              ("Deep Dive", ["Advanced concepts", "Best practices", "Real-world case studies"]),
                              ("Project Work", ["Portfolio project", "Peer review", "Optimisation"]),
                              ("Specialise", ["Pick a niche", "Community contribution"])],
             "Advanced": [("Expert Concepts", ["Cutting-edge topics", "Research papers"]),
                          ("Specialisation", ["Chosen niche in depth", "Tooling & automation"]),
                          ("Capstone", ["Large original project", "Publish / present"]),
                          ("Mastery & Teaching", ["Mentor others", "Write about it"])]}[level]
    weights = np.array([1.0] * len(names))
    weeks = np.maximum(1, np.round(weights / weights.sum() * total_weeks)).astype(int)
    kinds = ["Course", "Video", "Practice", "Book"]
    phases = []
    for i, (name, topics) in enumerate(names):
        phases.append({"name": name, "weeks": int(weeks[i]),
                       "topics": [f"{field}: {t}" for t in topics],
                       "milestone": f"Complete all '{name}' topics and pass a self-check quiz.",
                       "resources": [{"title": f"{field} {name}", "type": kinds[i % 4]},
                                     {"title": f"{field} {topics[0]} tutorial", "type": "Video"}]})
    return {"title": f"{field} Roadmap ({level})",
            "summary": f"A {int(weeks.sum())}-week plan at about {hours} h/week.",
            "total_weeks": int(weeks.sum()), "phases": phases}


# ---- Topic catalogue for ML recommendations ---------------------------------
CATALOGUE = {
    "Python Programming": "python syntax variables functions loops programming beginner coding",
    "Data Structures": "arrays lists trees graphs stacks queues algorithms programming",
    "Algorithms": "sorting searching complexity recursion dynamic programming algorithms",
    "Machine Learning": "regression classification training models data prediction supervised learning",
    "Deep Learning": "neural networks backpropagation layers training gradient models",
    "Statistics": "probability distribution mean variance hypothesis testing data",
    "Linear Algebra": "vectors matrices eigenvalues transformations mathematics",
    "Calculus": "derivatives integrals limits mathematics functions",
    "Databases & SQL": "sql tables queries joins relational data storage",
    "Web Development": "html css javascript browser frontend backend web",
    "Cybersecurity": "encryption security networks attacks defence passwords",
    "Computer Networks": "tcp ip protocols routing internet packets networks",
    "Operating Systems": "processes threads memory scheduling kernel computers",
    "Physics Basics": "motion force energy mechanics science physics",
    "Chemistry Basics": "atoms molecules reactions elements science chemistry",
    "Biology Basics": "cells genetics evolution organisms science biology",
    "World History": "civilizations empires wars revolutions history",
    "Economics": "markets supply demand inflation money economics",
    "Personal Finance": "budget saving investing interest money finance",
    "Public Speaking": "communication confidence presentation speech skills",
    "Time Management": "productivity planning focus habits skills study",
    "Data Visualization": "charts graphs plotting data dashboards pandas matplotlib",
    "Natural Language Processing": "text language tokens models transformers data",
    "Cloud Computing": "servers storage deployment scaling cloud infrastructure",
    "English Writing": "grammar essays writing vocabulary communication",
}


def recommend_topics(k=5):
    """Content-based recommender: cosine similarity between a user profile and the catalogue."""
    data = load_data()
    studied = [t["topic"] for t in data["completed_topics"]] + [e["topic"] for e in data["exams"]]
    df = exams_dataframe()
    weak = []
    if not df.empty:
        avg = df.groupby("topic")["percent"].mean()
        weak = list(avg[avg < 60].index)
    studied_l = {s.lower() for s in studied}
    names = [n for n in CATALOGUE if n.lower() not in studied_l]
    if not studied:
        return [(n, None) for n in list(CATALOGUE)[:k]]
    profile = " ".join(studied + weak * 2)                    # weak topics weigh double
    docs = [f"{n} {CATALOGUE[n]}" for n in names]
    vec = TfidfVectorizer(stop_words="english")
    m = vec.fit_transform(docs + [profile])
    sims = cosine_similarity(m[-1], m[:-1]).ravel()
    order = np.argsort(sims)[::-1][:k]
    return [(names[i], float(sims[i])) for i in order]


# =============================================================================
# 5. DOCUMENT EXPORT  (python-docx + FPDF2)
# Documents are described as "blocks": ("h1"|"h2"|"p"|"bullet", text)
# =============================================================================
def notes_blocks(n):
    b = [("h1", n["title"])]
    for s in n.get("sections", []):
        b.append(("h2", s["heading"]))
        b += [("bullet", p) for p in s["points"]]
    if n.get("definitions"):
        b.append(("h2", "Definitions"))
        b += [("bullet", f"{d['term']}: {d['meaning']}") for d in n["definitions"]]
    if n.get("examples"):
        b.append(("h2", "Examples"))
        b += [("bullet", e) for e in n["examples"]]
    if n.get("key_points"):
        b.append(("h2", "Key Points"))
        b += [("bullet", k) for k in n["key_points"]]
    return b


def summary_blocks(s, title="Lecture Summary"):
    b = [("h1", title), ("h2", "Short Summary"), ("p", s["short"]),
         ("h2", "Detailed Summary")]
    b += [("p", para) for para in s["detailed"].split("\n\n") if para.strip()]
    b.append(("h2", "Important Points"))
    b += [("bullet", k) for k in s["key_points"]]
    return b


def roadmap_blocks(r):
    b = [("h1", r["title"]), ("p", r.get("summary", ""))]
    for i, p in enumerate(r["phases"], 1):
        b.append(("h2", f"Phase {i}: {p['name']} ({p['weeks']} weeks)"))
        b += [("bullet", t) for t in p["topics"]]
        b.append(("p", f"Milestone: {p['milestone']}"))
        b += [("bullet", f"Resource: {x['title']} [{x['type']}]") for x in p.get("resources", [])]
    return b


def exam_report_blocks(topic, difficulty, score, total, questions, answers):
    b = [("h1", f"Exam Report - {topic}"),
         ("p", f"Difficulty: {difficulty} | Score: {score}/{total} ({100 * score / total:.1f}%)")]
    for i, q in enumerate(questions):
        chosen = answers.get(i)
        ok = chosen == q["answer"]
        b.append(("h2", f"Q{i + 1}. {q['question']}"))
        b.append(("p", f"Your answer: {q['options'][chosen] if chosen is not None else 'Not answered'} "
                       f"({'Correct' if ok else 'Incorrect'})"))
        if not ok:
            b.append(("p", f"Correct answer: {q['options'][q['answer']]}"))
            b.append(("p", f"Explanation: {q['explanation']}"))
    return b


def build_docx(blocks):
    doc = Document()
    for kind, text in blocks:
        if kind == "h1":
            doc.add_heading(text, 0)
        elif kind == "h2":
            doc.add_heading(text, 2)
        elif kind == "bullet":
            doc.add_paragraph(text, style="List Bullet")
        else:
            doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def pdf_safe(text):
    for a, b in {"\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"',
                 "\u201d": '"', "\u2022": "-", "\u2026": "...", "\u00a0": " "}.items():
        text = text.replace(a, b)
    text = text.encode("latin-1", "replace").decode("latin-1")
    # break absurdly long unbroken strings so FPDF can wrap them
    return re.sub(r"(\S{60})(?=\S)", r"\1 ", text)


def build_pdf(blocks):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for kind, text in blocks:
        text = pdf_safe(text)
        pdf.set_x(pdf.l_margin)
        if kind == "h1":
            pdf.set_font("Helvetica", "B", 20)
            pdf.set_text_color(79, 70, 229)
            pdf.multi_cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif kind == "h2":
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(124, 58, 237)
            pdf.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 6, ("- " + text) if kind == "bullet" else text,
                           new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
    return bytes(pdf.output())


def download_buttons(blocks, filename, key):
    c1, c2 = st.columns(2)
    try:
        c1.download_button("⬇️ Download Word (.docx)", build_docx(blocks), f"{filename}.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           key=f"{key}_docx")
        c2.download_button("⬇️ Download PDF", build_pdf(blocks), f"{filename}.pdf",
                           "application/pdf", key=f"{key}_pdf")
    except Exception as e:
        st.error(f"Could not build the download files: {e}")


# =============================================================================
# 6. CHARTS  (Plotly + Matplotlib + Seaborn)
# =============================================================================
def chart_score_trend(df):
    fig = px.line(df, x="date", y="percent", markers=True, hover_data=["topic", "difficulty"],
                  title="Exam scores over time (Plotly)")
    fig.update_traces(line=dict(color="#7c3aed", width=3), marker=dict(size=10))
    fig.update_layout(yaxis_range=[0, 105], yaxis_title="Score %", xaxis_title="",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def chart_progress_gauge(done, total):
    pct = 100 * done / total if total else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct, number={"suffix": "%"},
        title={"text": "Roadmap progress (Plotly)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#7c3aed"},
               "steps": [{"range": [0, 50], "color": "#e0e7ff"}, {"range": [50, 100], "color": "#c7d2fe"}]}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def chart_avg_by_difficulty(df):
    avg = df.groupby("difficulty")["percent"].mean().reindex(["Easy", "Medium", "Hard", "Mixed"]).dropna()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(avg.index, avg.values, color=["#22c55e", "#f59e0b", "#ef4444", "#7c3aed"][:len(avg)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Average score %")
    ax.set_title("Average score by difficulty (Matplotlib)")
    for i, v in enumerate(avg.values):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center")
    fig.tight_layout()
    return fig


def chart_activity_heatmap(activity):
    df = pd.DataFrame(activity)
    cutoff = (date.today() - timedelta(days=13)).isoformat()
    df = df[df["date"] >= cutoff]
    if df.empty:
        return None
    pivot = df.pivot_table(index="type", columns="date", values="detail", aggfunc="count", fill_value=0)
    pivot.columns = [c[5:] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Purples", cbar=False, linewidths=.5, ax=ax)
    ax.set_title("Activity - last 14 days (Seaborn)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def chart_score_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    sns.histplot(df["percent"], bins=np.arange(0, 110, 10), color="#4f46e5", ax=ax)
    ax.set_xlabel("Score %")
    ax.set_title("Score distribution (Seaborn)")
    fig.tight_layout()
    return fig


FIREWORKS_HTML = """
<div style="position:relative;width:100%;height:430px;background:radial-gradient(circle at 50% 100%,#1e1b4b,#020617);border-radius:20px;overflow:hidden;font-family:Poppins,sans-serif;">
<canvas id="c" style="position:absolute;inset:0;width:100%;height:100%"></canvas>
<div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:white;padding:20px;">
  <div style="font-size:58px;animation:pop 1s ease infinite alternate">🎉</div>
  <h1 style="margin:6px 0;font-size:2.1rem;background:linear-gradient(90deg,#fde68a,#f9a8d4,#93c5fd);-webkit-background-clip:text;color:transparent;">__TITLE__</h1>
  <h2 style="margin:4px 0;font-weight:500">You scored __SCORE__</h2>
  <p style="max-width:520px;opacity:.92;font-size:1.05rem">__MESSAGE__</p>
</div></div>
<style>@keyframes pop{from{transform:scale(1)}to{transform:scale(1.18)}}</style>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
function resize(){cv.width=cv.offsetWidth;cv.height=cv.offsetHeight}resize();
let ps=[];
function burst(x,y){const h=Math.random()*360;for(let i=0;i<70;i++){const a=Math.random()*6.283,s=Math.random()*5+1.5;
ps.push({x,y,vx:Math.cos(a)*s,vy:Math.sin(a)*s,life:1,hue:h+Math.random()*40});}}
function loop(){ctx.fillStyle='rgba(2,6,23,0.22)';ctx.fillRect(0,0,cv.width,cv.height);
if(Math.random()<0.06)burst(Math.random()*cv.width*0.8+cv.width*0.1,Math.random()*cv.height*0.55+20);
ps=ps.filter(p=>p.life>0.02);
for(const p of ps){p.x+=p.vx;p.y+=p.vy;p.vy+=0.04;p.vx*=0.985;p.life*=0.975;
ctx.beginPath();ctx.fillStyle=`hsla(${p.hue},100%,60%,${p.life})`;ctx.arc(p.x,p.y,2.2,0,6.283);ctx.fill();}
requestAnimationFrame(loop)}
burst(cv.width/2,cv.height/3);loop();
</script>
"""


def show_celebration(score, total):
    pct = 100 * score / total if total else 0
    if pct >= 90:
        title, msg = "Outstanding Performance!", "You've truly mastered this topic. Your hard work is shining through!"
    elif pct >= 70:
        title, msg = "Great Job!", "Excellent effort! You're building strong knowledge - keep up the momentum."
    elif pct >= 50:
        title, msg = "Well Done!", "You're on the right track. Review the explanations below and you'll improve fast."
    else:
        title, msg = "You Showed Up - That Counts!", "Every expert started here. Learn from the explanations and try again - you've got this!"
    html = (FIREWORKS_HTML.replace("__TITLE__", title)
            .replace("__SCORE__", f"{score}/{total} ({pct:.0f}%)").replace("__MESSAGE__", msg))
    components.html(html, height=440)
    if pct >= 70:
        st.balloons()


# =============================================================================
# 7. PAGES  (Streamlit)
# =============================================================================
def read_upload(file):
    """Read text from .txt / .md / .pdf / .docx uploads."""
    name = file.name.lower()
    try:
        if name.endswith((".txt", ".md")):
            return file.read().decode("utf-8", errors="ignore")
        if name.endswith(".pdf"):
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(file).pages)
        if name.endswith(".docx"):
            return "\n".join(p.text for p in Document(file).paragraphs)
    except Exception as e:
        st.error(f"Could not read '{file.name}': {e}")
        return ""
    st.error("Unsupported file type. Use .txt, .md, .pdf or .docx.")
    return ""


def page_home():
    hero("🌧️ Arora Rain", "Your personal AI learning companion - learn, summarise, practise and grow.")
    data = load_data()
    df = exams_dataframe()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topics completed", len(data["completed_topics"]))
    c2.metric("Exams taken", len(df))
    c3.metric("Average score", f"{df['percent'].mean():.0f}%" if not df.empty else "-")
    c4.metric("Day streak 🔥", current_streak())
    st.write("")
    feats = [("🧠 AI Learning Assistant", "Ask about any topic and get clear explanations with examples."),
             ("📝 Lecture Summarizer", "Short and detailed summaries with the key points highlighted."),
             ("📒 Smart Notes", "Structured notes you can download as PDF or Word."),
             ("🎯 AI Exam Generator", "10-100 MCQs, difficulty levels, marks and explanations."),
             ("🗺️ Learning Roadmap", "A personalised plan with milestones and resources."),
             ("📊 Progress Dashboard", "Track topics, scores and activity with live charts.")]
    cols = st.columns(3)
    for i, (t, d) in enumerate(feats):
        with cols[i % 3]:
            card(t, f"<p style='margin:0;opacity:.85'>{d}</p>")
    st.markdown("#### 🎁 Recommended for you")
    for name, sim in recommend_topics():
        tag = f" <span class='pill'>match {sim * 100:.0f}%</span>" if sim is not None else ""
        st.markdown(f"- **{name}**{tag}", unsafe_allow_html=True)
    if not ai_enabled():
        st.info("Tip: add your Claude API key in the sidebar for AI-generated content. "
                "Without it, Arora Rain uses its offline engine (Wikipedia + machine learning).")


def page_assistant():
    hero("🧠 AI Learning Assistant", "Enter any topic and get an easy explanation with examples.")
    c1, c2 = st.columns([3, 1])
    topic = c1.text_input("What do you want to learn?", placeholder="e.g. Photosynthesis, Recursion, Inflation")
    level = c2.selectbox("Explain for", ["Beginner", "Intermediate", "Advanced"])
    if st.button("✨ Explain it"):
        topic = topic.strip()
        if len(topic) < 2:
            st.error("Please enter a topic (at least 2 characters).")
        else:
            with st.spinner("Thinking..."):
                try:
                    def offline():
                        md, _ = offline_explanation(topic)
                        return md
                    result, mode = with_fallback(lambda: ai_explain(topic, level), offline, "explanations")
                    st.session_state["explanation"] = {"topic": topic, "text": result, "mode": mode}
                    log_activity("Explain", topic)
                except (ValueError, AIError) as e:
                    st.error(str(e))
    exp = st.session_state.get("explanation")
    if exp:
        st.markdown(f"### 📖 {exp['topic']}  <span class='pill'>{exp['mode'].upper()}</span>",
                    unsafe_allow_html=True)
        st.markdown(exp["text"])
        c1, c2 = st.columns(2)
        if c1.button("✅ Mark topic as studied"):
            mark_completed(exp["topic"], "assistant")
            st.success("Added to your completed topics!")
        with c2:
            pass
        blocks = [("h1", exp["topic"])] + [
            ("h2" if l.startswith("##") else "bullet" if l.lstrip().startswith(("-", "*")) else "p",
             re.sub(r"^[#\-*\s]+", "", l).replace("**", ""))
            for l in exp["text"].splitlines() if l.strip()]
        download_buttons(blocks, re.sub(r"\W+", "_", exp["topic"]), "explain")


def get_source_text(key):
    """Text area + file uploader used by summarizer and notes pages."""
    text = st.text_area("Paste lecture text", height=220, key=f"{key}_text",
                        placeholder="Paste your lecture notes or transcript here...")
    up = st.file_uploader("...or upload a document (.txt, .md, .pdf, .docx)",
                          type=["txt", "md", "pdf", "docx"], key=f"{key}_up")
    if up is not None:
        extracted = read_upload(up)
        if extracted.strip():
            st.caption(f"Loaded {len(extracted.split())} words from {up.name}")
            text = extracted
    return text.strip()


def page_summarizer():
    hero("📝 Lecture Summarizer", "Turn long lectures into short and detailed summaries.")
    text = get_source_text("sum")
    if st.button("Summarise"):
        if len(text.split()) < 40:
            st.error("Please provide at least 40 words of text.")
        else:
            with st.spinner("Summarising..."):
                try:
                    res, mode = with_fallback(lambda: ai_summary(text), lambda: offline_summary(text), "summaries")
                    st.session_state["summary"] = res
                    log_activity("Summarize", f"{len(text.split())} words")
                except (ValueError, AIError) as e:
                    st.error(str(e))
    s = st.session_state.get("summary")
    if s:
        t1, t2, t3 = st.tabs(["⚡ Short", "📄 Detailed", "⭐ Key points"])
        t1.write(s["short"])
        t2.write(s["detailed"])
        with t3:
            for k in s["key_points"]:
                st.markdown(f"- **{k}**")
        download_buttons(summary_blocks(s), "lecture_summary", "sum")


def page_notes():
    hero("📒 Smart Notes Generator", "Convert lecture content into organised, downloadable notes.")
    title = st.text_input("Notes title", "My Study Notes")
    mode = st.radio("Source", ["Paste / upload text", "Generate from a topic"], horizontal=True)
    text, topic = "", ""
    if mode == "Paste / upload text":
        text = get_source_text("notes")
    else:
        topic = st.text_input("Topic", placeholder="e.g. Machine Learning")
    if st.button("Create notes"):
        try:
            if mode == "Paste / upload text":
                if len(text.split()) < 40:
                    raise ValueError("Please provide at least 40 words of text.")
                notes, _ = with_fallback(lambda: ai_notes(text, title),
                                         lambda: offline_notes(text, title), "notes")
            else:
                if len(topic.strip()) < 2:
                    raise ValueError("Please enter a topic.")
                def ai():
                    return ai_notes(call_llm(f"Write a detailed lecture on: {topic}", max_tokens=3000),
                                    title or topic)

                def offline():
                    page = fetch_wikipedia(topic.strip())
                    if not page:
                        raise ValueError("No offline source found for that topic.")
                    return offline_notes(" ".join(s["text"] for s in page["sections"]),
                                         title or page["title"], page["sections"])
                notes, _ = with_fallback(ai, offline, "notes")
            st.session_state["notes"] = notes
            log_activity("Notes", notes["title"])
        except (ValueError, AIError) as e:
            st.error(str(e))
    n = st.session_state.get("notes")
    if n:
        st.markdown(f"## {n['title']}")
        for s in n["sections"]:
            st.markdown(f"### {s['heading']}")
            for p in s["points"]:
                st.markdown(f"- {p}")
        if n["definitions"]:
            st.markdown("### 📚 Definitions")
            for d in n["definitions"]:
                st.markdown(f"- **{d['term']}** - {d['meaning']}")
        if n["examples"]:
            st.markdown("### 💡 Examples")
            for e in n["examples"]:
                st.markdown(f"- {e}")
        st.markdown("### ⭐ Key points")
        for k in n["key_points"]:
            st.markdown(f"- {k}")
        download_buttons(notes_blocks(n), re.sub(r"\W+", "_", n["title"]), "notes")


def page_exam():
    hero("🎯 AI Exam Generator", "Practise with custom MCQs, get marks and explanations.")
    ex = st.session_state.get("exam")

    if ex is None:
        with st.form("exam_setup"):
            topic = st.text_input("Exam topic", placeholder="e.g. Solar System, Python, World War II")
            c1, c2 = st.columns(2)
            difficulty = c1.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Mixed"])
            n = c2.slider("Number of questions", 10, 100, 10, step=5)
            go_btn = st.form_submit_button("🚀 Generate exam")
        if go_btn:
            if len(topic.strip()) < 2:
                st.error("Please enter a topic.")
                return
            with st.spinner(f"Creating {n} questions..."):
                try:
                    qs, mode = with_fallback(lambda: ai_exam(topic.strip(), difficulty, n),
                                             lambda: offline_exam(topic.strip(), difficulty, n), "exams")
                except (ValueError, AIError) as e:
                    st.error(str(e))
                    return
            if len(qs) < n:
                st.warning(f"Only {len(qs)} questions could be generated from the available source text.")
            st.session_state["exam"] = {"id": uuid.uuid4().hex[:8], "topic": topic.strip(),
                                        "difficulty": difficulty, "questions": qs,
                                        "submitted": False, "mode": mode}
            st.rerun()
        return

    qs = ex["questions"]
    st.markdown(f"### {ex['topic']}  <span class='pill'>{ex['difficulty']}</span>"
                f"<span class='pill'>{len(qs)} questions</span><span class='pill'>{ex['mode'].upper()}</span>",
                unsafe_allow_html=True)

    if not ex["submitted"]:
        with st.form("exam_form"):
            for i, q in enumerate(qs):
                st.markdown(f"**Q{i + 1}. {q['question']}**  <span class='pill'>{q['difficulty']}</span>",
                            unsafe_allow_html=True)
                st.radio("Choose one", q["options"], index=None, key=f"ans_{ex['id']}_{i}",
                         label_visibility="collapsed")
                st.divider()
            submitted = st.form_submit_button("✅ Submit exam")
        if submitted:
            answers, score = {}, 0
            for i, q in enumerate(qs):
                chosen = st.session_state.get(f"ans_{ex['id']}_{i}")
                if chosen is not None:
                    answers[i] = q["options"].index(chosen)
                    score += int(answers[i] == q["answer"])
            ex.update(submitted=True, answers=answers, score=score)
            record_exam(ex["topic"], ex["difficulty"], score, len(qs))
            log_activity("Exam", ex["topic"])
            if score / len(qs) >= 0.6:
                mark_completed(ex["topic"], "exam")
            st.rerun()
    else:
        score, total = ex["score"], len(qs)
        show_celebration(score, total)
        c1, c2, c3 = st.columns(3)
        c1.metric("Marks", f"{score}/{total}")
        c2.metric("Percentage", f"{100 * score / total:.1f}%")
        c3.metric("Wrong / skipped", total - score)
        st.markdown("### 🔍 Review")
        wrong = 0
        for i, q in enumerate(qs):
            chosen = ex["answers"].get(i)
            if chosen == q["answer"]:
                continue
            wrong += 1
            with st.expander(f"❌ Q{i + 1}. {q['question']}"):
                st.write(f"**Your answer:** {q['options'][chosen] if chosen is not None else 'Not answered'}")
                st.write(f"**Correct answer:** {q['options'][q['answer']]}")
                st.info(q["explanation"] or "No explanation provided.")
        if wrong == 0:
            st.success("Perfect score - nothing to review! 🌟")
        download_buttons(exam_report_blocks(ex["topic"], ex["difficulty"], score, total, qs, ex["answers"]),
                         "exam_report", "exam")
        if st.button("🔄 Start a new exam"):
            st.session_state["exam"] = None
            st.rerun()


def toggle_topic(topic, key):
    if st.session_state.get(key):
        mark_completed(topic, "roadmap")
    else:
        unmark_completed(topic)


def page_roadmap():
    hero("🗺️ Personalised Learning Roadmap", "Tell us your goal - we'll plan the journey.")
    with st.form("roadmap_form"):
        field = st.text_input("Learning field", placeholder="e.g. Data Science, Web Design, Biology")
        c1, c2 = st.columns(2)
        level = c1.selectbox("Skill level", ["Beginner", "Intermediate", "Advanced"])
        hours = c2.slider("Available hours per week", 1, 40, 6)
        make = st.form_submit_button("🧭 Build my roadmap")
    if make:
        if len(field.strip()) < 2:
            st.error("Please enter your learning field.")
        else:
            with st.spinner("Designing your roadmap..."):
                rm, mode = with_fallback(lambda: ai_roadmap(field.strip(), level, hours),
                                         lambda: offline_roadmap(field.strip(), level, hours), "roadmaps")
            rm.update(field=field.strip(), level=level, hours=hours)
            data = load_data()
            data["roadmap"] = rm
            save_data(data)
            log_activity("Roadmap", field.strip())
    rm = load_data().get("roadmap")
    if not rm:
        st.info("Fill in the form above to generate your roadmap.")
        return
    st.markdown(f"## {rm['title']}")
    st.caption(rm.get("summary", ""))
    done_set = {t["topic"].lower() for t in load_data()["completed_topics"]}
    for i, p in enumerate(rm["phases"]):
        with st.expander(f"Phase {i + 1}: {p['name']}  -  {p['weeks']} week(s)", expanded=(i == 0)):
            for j, t in enumerate(p["topics"]):
                key = f"rm_{i}_{j}"
                st.checkbox(t, value=t.lower() in done_set, key=key, on_change=toggle_topic, args=(t, key))
            st.markdown(f"🏁 **Milestone:** {p['milestone']}")
            for r in p.get("resources", []):
                link = resource_link(r["title"], r.get("type", ""), rm.get("field", ""))
                st.markdown(f"- [{r['title']}]({link})  <span class='pill'>{r.get('type', '')}</span>",
                            unsafe_allow_html=True)
    download_buttons(roadmap_blocks(rm), "learning_roadmap", "rm")


def page_dashboard():
    hero("📊 Progress Dashboard", "See how far you've come.")
    data = load_data()
    df = exams_dataframe()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topics completed", len(data["completed_topics"]))
    c2.metric("Exams taken", len(df))
    c3.metric("Best score", f"{df['percent'].max():.0f}%" if not df.empty else "-")
    c4.metric("Day streak 🔥", current_streak())

    left, right = st.columns(2)
    rm = data.get("roadmap")
    with left:
        if rm:
            topics = [t for p in rm["phases"] for t in p["topics"]]
            done = {t["topic"].lower() for t in data["completed_topics"]}
            st.plotly_chart(chart_progress_gauge(sum(t.lower() in done for t in topics), len(topics)),
                            use_container_width=True)
        else:
            st.info("Create a roadmap to see your progress gauge.")
    with right:
        if not df.empty:
            st.plotly_chart(chart_score_trend(df), use_container_width=True)
        else:
            st.info("Take an exam to see your score trend.")

    if not df.empty:
        a, b = st.columns(2)
        a.pyplot(chart_avg_by_difficulty(df))
        b.pyplot(chart_score_distribution(df))
    if data["activity"]:
        fig = chart_activity_heatmap(data["activity"])
        if fig:
            st.pyplot(fig)

    st.markdown("### ✅ Completed topics")
    if data["completed_topics"]:
        st.dataframe(pd.DataFrame(data["completed_topics"]), use_container_width=True, hide_index=True)
    else:
        st.caption("Nothing yet - study a topic and mark it as completed!")
    st.markdown("### 📝 Exam history")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("### 🎁 Recommended next topics (Scikit-learn)")
    for name, sim in recommend_topics():
        st.markdown(f"- **{name}**" + (f" - similarity {sim * 100:.0f}%" if sim is not None else ""))
    with st.expander("⚠️ Danger zone"):
        if st.button("Reset all progress"):
            save_data(default_data())
            st.success("Progress cleared.")
            st.rerun()


# =============================================================================
# MAIN: sidebar navigation
# =============================================================================
PAGES = {"🏠 Home": page_home, "🧠 AI Assistant": page_assistant, "📝 Summarizer": page_summarizer,
         "📒 Smart Notes": page_notes, "🎯 Exam Generator": page_exam,
         "🗺️ Roadmap": page_roadmap, "📊 Dashboard": page_dashboard}


def main():
    with st.sidebar:
        st.markdown("## 🌧️ Arora Rain")
        st.caption("AI Learning Companion")
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.markdown("**🔑 AI settings**")
        st.text_input("Claude API key", type="password", key="api_key",
                      help="Stored only in this browser session. You can also set ANTHROPIC_API_KEY.")
        st.text_input("Model", value=DEFAULT_MODEL, key="model")
        st.caption("AI mode: ✅ ON" if ai_enabled() else "AI mode: ⚪ offline engine")
    try:
        PAGES[choice]()
    except Exception as e:                          # last-resort safety net
        st.error(f"Something went wrong: {e}")


if __name__ == "__main__":
    main()
