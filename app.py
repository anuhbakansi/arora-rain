"""
ARORA RAIN - AI Learning Companion
===================================
Run with:  streamlit run app.py

Libraries used (and nothing else besides the Python standard library):
  * Streamlit   -> the whole interface: sidebar, forms, tabs, uploads, downloads, animations (CSS/HTML)
  * Pandas      -> learning records (exams, topics, activity) as DataFrames, grouping and pivots
  * NumPy       -> statistics: averages, trend line, consistency, streak helpers
  * Matplotlib  -> educational charts (progress donut, difficulty bars, topic performance)
  * Seaborn     -> statistical charts (score trend, distribution, activity heatmap)
  * google-genai (official Google Gen AI SDK) -> every AI feature, using a Gemini API key

The API key is NEVER hard-coded or typed into the page. It is read from
Streamlit secrets (.streamlit/secrets.toml) or an environment variable.

File layout (sections of this single file):
  1. CONFIG & STYLING   2. STORAGE   3. GEMINI AI ENGINE   4. AI FEATURES (prompts)
  5. EXPORT HELPERS     6. CHARTS    7. PAGES              8. MAIN (sidebar navigation)
"""

import json
import os
import random
import re
import time
import uuid
from datetime import date, datetime, timedelta
from urllib.parse import quote_plus

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

# =============================================================================
# 1. CONFIG & STYLING  (Streamlit)
# =============================================================================
APP_NAME = "Arora Rain"
DATA_FILE = "arora_data.json"
DEFAULT_MODEL = "gemini-2.5-flash"

st.set_page_config(page_title=f"{APP_NAME} - AI Learning Companion", page_icon="🌧️", layout="wide")

# The Google Gen AI SDK is required; show a friendly message if it is missing.
try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError:
    st.error("The Google Gen AI SDK is not installed. Run:  `pip install -r requirements.txt`  "
             "(or `pip install google-genai`) and restart the app.")
    st.stop()

PRIMARY, SECONDARY, ACCENT = "#4f46e5", "#7c3aed", "#06b6d4"
PALETTE = [PRIMARY, SECONDARY, ACCENT, "#22c55e", "#f59e0b", "#ef4444"]

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
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: white; }
.hero p { margin: 6px 0 0; opacity: .92; font-size: 1.02rem; color: white; }
.card {
  background: rgba(124,58,237,.06); border: 1px solid rgba(124,58,237,.25);
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
sns.set_theme(style="whitegrid", font="DejaVu Sans")


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


# =============================================================================
# 2. STORAGE  (JSON file + Pandas)
# =============================================================================
def default_data():
    return {"completed_topics": [], "exams": [], "activity": [], "roadmap": None}


def load_data():
    """Load saved progress. A missing or corrupted file gives fresh data."""
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
        data["completed_topics"].append({"topic": topic, "source": source, "date": date.today().isoformat()})
        save_data(data)


def unmark_completed(topic):
    data = load_data()
    data["completed_topics"] = [t for t in data["completed_topics"] if t["topic"].lower() != topic.lower()]
    save_data(data)


def record_exam(topic, difficulty, score, total):
    data = load_data()
    data["exams"].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "topic": topic,
                          "difficulty": difficulty, "score": score, "total": total,
                          "percent": round(100 * score / total, 1) if total else 0.0})
    save_data(data)


def exams_df():
    """Exam history as a Pandas DataFrame (empty but well-formed when there is no data)."""
    exams = load_data()["exams"]
    cols = ["date", "topic", "difficulty", "score", "total", "percent"]
    return pd.DataFrame(exams, columns=cols) if exams else pd.DataFrame(columns=cols)


def current_streak():
    dates = sorted({a["date"] for a in load_data()["activity"]}, reverse=True)
    if not dates:
        return 0
    day, streak = date.today(), 0
    if dates[0] != day.isoformat():
        day -= timedelta(days=1)                      # streak may continue from yesterday
    for d in dates:
        if d == day.isoformat():
            streak += 1
            day -= timedelta(days=1)
        elif d < day.isoformat():
            break
    return streak


# =============================================================================
# 3. GEMINI AI ENGINE  (official Google Gen AI SDK)
# =============================================================================
class AIError(Exception):
    """Any problem talking to Gemini, with a message that is safe to show the user."""


def get_api_key():
    """Read the key from Streamlit secrets or environment variables - never from code or the UI."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            val = st.secrets.get(name, "")
        except Exception:                             # no secrets.toml present
            val = ""
        val = val or os.environ.get(name, "")
        val = re.sub(r"\s+", "", str(val)).strip("\"'`")
        if val:
            return val
    return ""


def get_model():
    name = st.session_state.get("model") or ""
    if not name:
        try:
            name = st.secrets.get("GEMINI_MODEL", "")
        except Exception:
            name = ""
    return (name or os.environ.get("GEMINI_MODEL", "") or DEFAULT_MODEL).strip()


@st.cache_resource(show_spinner=False)
def _client(api_key):
    return genai.Client(api_key=api_key)


def friendly_error(code, message):
    text = str(message)
    low = text.lower()
    if code in (400, 401, 403) and ("api key" in low or "api_key" in low or code in (401, 403)):
        return ("Google rejected the API key. Create a new key at aistudio.google.com -> Get API key, "
                "update GEMINI_API_KEY and restart the app.")
    if code == 404:
        return f"Model '{get_model()}' was not found. Change the Model name in the sidebar."
    if code == 429:
        return "Free-tier rate limit reached. Wait a minute and try again (or request fewer questions)."
    if code in (500, 503):
        return "Gemini is busy right now. Please try again in a moment."
    return f"Gemini error {code}: {text[:300]}"


def generate(prompt, system="You are Arora Rain, a friendly and accurate expert teacher.",
             json_mode=False, temperature=0.5):
    """Call Gemini through the official SDK. Retries briefly on rate-limit / overload errors."""
    key = get_api_key()
    if not key:
        raise AIError("No Gemini API key found. See the setup steps on the Home page.")
    kwargs = {"system_instruction": system, "temperature": temperature}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    config = genai_types.GenerateContentConfig(**kwargs)
    for attempt in range(3):
        try:
            resp = _client(key).models.generate_content(model=get_model(), contents=prompt, config=config)
            text = resp.text
            if not text or not text.strip():
                raise AIError("Gemini returned an empty answer (it may have been blocked). Try rephrasing.")
            return text
        except AIError:
            raise
        except genai_errors.APIError as e:
            code = getattr(e, "code", None)
            if code in (429, 500, 503) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise AIError(friendly_error(code, getattr(e, "message", e)))
        except Exception as e:                        # network problems etc.
            raise AIError(f"Could not reach Gemini: {e}")
    raise AIError("Gemini did not respond. Please try again.")


def parse_json(text):
    """Parse a JSON reply, tolerating code fences or extra text around it."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    end = max(text.rfind("}"), text.rfind("]"))
    if starts and end > min(starts):
        try:
            return json.loads(text[min(starts):end + 1])
        except json.JSONDecodeError:
            pass
    raise AIError("The AI reply was not valid JSON. Please try again.")


def require_ai():
    """Show setup help and return False when no key is configured."""
    if get_api_key():
        return True
    st.warning("🔑 Gemini API key not found. Follow the 3 steps below, then restart the app.")
    st.markdown(SETUP_HELP)
    return False


SETUP_HELP = """
1. Create a free key at **https://aistudio.google.com** -> *Get API key*.
2. Create the file `.streamlit/secrets.toml` next to `app.py` and add:
   ```toml
   GEMINI_API_KEY = "paste-your-key-here"
   ```
   *(or set the environment variable `GEMINI_API_KEY` before starting the app)*
3. Restart: `streamlit run app.py`
"""


# =============================================================================
# 4. AI FEATURES  (prompts -> structured results)
# =============================================================================
def ai_explain(topic, level):
    prompt = (f"Explain '{topic}' in simple language for a {level} learner. Use Markdown with exactly these "
              "sections: ## Simple Explanation, ## Key Concepts (bullet list), ## Examples (2-3 concrete "
              "examples), ## Common Mistakes, ## Quick Recap. Be clear, friendly and accurate.")
    return generate(prompt, temperature=0.6)


def ai_summary(text):
    prompt = ("Summarise the lecture text below. Reply with JSON: "
              '{"short": "3-4 sentence summary", "detailed": "a detailed multi-paragraph summary '
              '(separate paragraphs with a blank line)", "key_points": ["important point", ...]} '
              "with 5-8 key points.\n\nLECTURE TEXT:\n" + text[:30000])
    d = parse_json(generate(prompt, json_mode=True, temperature=0.3))
    if not isinstance(d, dict):
        raise AIError("Unexpected summary format. Please try again.")
    return {"short": str(d.get("short", "")), "detailed": str(d.get("detailed", "")),
            "key_points": [str(k) for k in d.get("key_points", [])]}


NOTES_SHAPE = ('{"title": str, "sections": [{"heading": str, "points": [str]}], '
               '"definitions": [{"term": str, "meaning": str}], "examples": [str], "key_points": [str]}')


def ai_notes(source, title, from_topic):
    if from_topic:
        prompt = (f"Create well-organised study notes on the topic '{source}'. Reply with JSON shaped "
                  f"exactly like {NOTES_SHAPE}. Use 4-7 sections with clear headings, 4-8 definitions, "
                  "3-5 worked examples and 6-8 key points.")
    else:
        prompt = (f"Convert the lecture content below into organised study notes. Reply with JSON shaped "
                  f"exactly like {NOTES_SHAPE}. Use clear headings, definitions, examples (from the text or "
                  "invented to illustrate) and key points.\n\nLECTURE CONTENT:\n" + source[:30000])
    d = parse_json(generate(prompt, json_mode=True, temperature=0.4))
    if not isinstance(d, dict) or not d.get("sections"):
        raise AIError("Could not build notes from the AI reply. Please try again.")
    d["title"] = title or d.get("title") or "Study Notes"
    for k in ("definitions", "examples", "key_points"):
        d.setdefault(k, [])
    return d


def ai_exam(topic, difficulty, n, progress=None):
    """Generate n MCQs in batches. Options are shuffled so the answer position is not biased."""
    questions, seen, fails = [], [], 0
    while len(questions) < n:
        batch = min(20, n - len(questions))
        avoid = ("\nDo NOT repeat any of these questions: " + " | ".join(seen[-60:])) if seen else ""
        prompt = (f"Create {batch} multiple-choice questions about '{topic}'. Difficulty: {difficulty} "
                  "(for 'Mixed', vary between Easy, Medium and Hard). Reply with a JSON array of objects: "
                  '{"question": str, "options": [exactly 4 strings], "answer": 0-3 index of the correct '
                  'option, "explanation": str explaining why the answer is correct, '
                  '"difficulty": "Easy" | "Medium" | "Hard"}. Exactly one option must be correct.' + avoid)
        added = 0
        data = parse_json(generate(prompt, json_mode=True, temperature=0.8))
        if isinstance(data, dict):
            data = data.get("questions", [])
        for q in data if isinstance(data, list) else []:
            try:
                opts = [str(o).strip() for o in q["options"]][:4]
                ans = int(q["answer"])
                stem = str(q["question"]).strip()
                if len(opts) == 4 and len(set(opts)) == 4 and 0 <= ans < 4 and stem not in seen:
                    correct = opts[ans]
                    random.shuffle(opts)
                    questions.append({"question": stem, "options": opts, "answer": opts.index(correct),
                                      "explanation": str(q.get("explanation", "")).strip(),
                                      "difficulty": str(q.get("difficulty", difficulty)).title()})
                    seen.append(stem)
                    added += 1
            except (KeyError, ValueError, TypeError):
                continue
        fails = 0 if added else fails + 1
        if fails >= 2:
            raise AIError("Gemini could not produce valid questions for that topic. Try a clearer topic.")
        if progress:
            progress(min(len(questions), n) / n)
    return questions[:n]


def ai_roadmap(field, level, hours):
    prompt = (f"Create a personalised learning roadmap. Field: {field}. Skill level: {level}. Available time: "
              f"{hours} hours per week. Reply with JSON: "
              '{"title": str, "summary": str, "total_weeks": int, "phases": [{"name": str, "weeks": int, '
              '"topics": [3-6 short topic strings], "milestone": str, "resources": [{"title": str, '
              '"type": "Course" | "Book" | "Video" | "Practice" | "Docs"}]}]}. Use 4-6 phases ordered from '
              "easiest to hardest, realistic week counts for the available time, and real, well-known "
              "resource titles.")
    d = parse_json(generate(prompt, json_mode=True, temperature=0.5))
    if not isinstance(d, dict) or not d.get("phases"):
        raise AIError("Roadmap reply was incomplete. Please try again.")
    for p in d["phases"]:
        p["weeks"] = int(p.get("weeks", 1) or 1)
        p.setdefault("topics", [])
        p.setdefault("resources", [])
        p.setdefault("milestone", "")
    d.setdefault("title", f"{field} Roadmap")
    d.setdefault("summary", "")
    d["total_weeks"] = int(sum(p["weeks"] for p in d["phases"]))
    return d


def ai_recommendations(df, completed):
    weak = ""
    if not df.empty:
        avg = df.groupby("topic")["percent"].mean().round(0)
        weak = "; ".join(f"{t}: {v:.0f}%" for t, v in avg.items())
    prompt = ("Recommend 5 next topics for this learner. Completed topics: "
              f"{', '.join(completed) or 'none yet'}. Exam averages by topic: {weak or 'no exams yet'}. "
              'Reply with a JSON array: [{"topic": str, "reason": str (one sentence)}].')
    d = parse_json(generate(prompt, json_mode=True, temperature=0.6))
    return [x for x in d if isinstance(x, dict) and "topic" in x][:5] if isinstance(d, list) else []


# =============================================================================
# 5. EXPORT HELPERS  (plain Markdown / text downloads via Streamlit)
# Documents are lists of blocks: ("h1"|"h2"|"p"|"bullet", text)
# =============================================================================
def notes_blocks(n):
    b = [("h1", n["title"])]
    for s in n.get("sections", []):
        b.append(("h2", str(s.get("heading", ""))))
        b += [("bullet", str(p)) for p in s.get("points", [])]
    if n.get("definitions"):
        b.append(("h2", "Definitions"))
        b += [("bullet", f"{d.get('term', '')}: {d.get('meaning', '')}") for d in n["definitions"]]
    if n.get("examples"):
        b.append(("h2", "Examples"))
        b += [("bullet", str(e)) for e in n["examples"]]
    if n.get("key_points"):
        b.append(("h2", "Key Points"))
        b += [("bullet", str(k)) for k in n["key_points"]]
    return b


def summary_blocks(s):
    b = [("h1", "Lecture Summary"), ("h2", "Short Summary"), ("p", s["short"]), ("h2", "Detailed Summary")]
    b += [("p", p) for p in s["detailed"].split("\n\n") if p.strip()]
    b.append(("h2", "Key Points"))
    b += [("bullet", k) for k in s["key_points"]]
    return b


def roadmap_blocks(r):
    b = [("h1", r["title"]), ("p", r.get("summary", ""))]
    for i, p in enumerate(r["phases"], 1):
        b.append(("h2", f"Phase {i}: {p['name']} ({p['weeks']} weeks)"))
        b += [("bullet", t) for t in p["topics"]]
        b.append(("p", f"Milestone: {p['milestone']}"))
        b += [("bullet", f"Resource: {x.get('title', '')} [{x.get('type', '')}]") for x in p["resources"]]
    return b


def exam_blocks(topic, difficulty, score, total, questions, answers):
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


def blocks_to_markdown(blocks):
    prefix = {"h1": "# ", "h2": "\n## ", "bullet": "- ", "p": ""}
    return "\n".join(prefix[k] + t + ("\n" if k == "p" else "") for k, t in blocks)


def blocks_to_text(blocks):
    out = []
    for k, t in blocks:
        if k == "h1":
            out += [t.upper(), "=" * len(t), ""]
        elif k == "h2":
            out += ["", t, "-" * len(t)]
        elif k == "bullet":
            out.append(f"  * {t}")
        else:
            out += [t, ""]
    return "\n".join(out)


def download_buttons(blocks, filename, key):
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download Markdown (.md)", blocks_to_markdown(blocks), f"{filename}.md",
                       "text/markdown", key=f"{key}_md")
    c2.download_button("⬇️ Download Text (.txt)", blocks_to_text(blocks), f"{filename}.txt",
                       "text/plain", key=f"{key}_txt")


def safe_name(text):
    return re.sub(r"\W+", "_", text).strip("_")[:50] or "arora_rain"


# =============================================================================
# 6. CHARTS  (Matplotlib + Seaborn, data prepared with Pandas + NumPy)
# =============================================================================
def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def chart_score_trend(df):
    d = df.copy()
    d["attempt"] = np.arange(1, len(d) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sns.lineplot(data=d, x="attempt", y="percent", marker="o", color=PRIMARY, linewidth=2.5, ax=ax)
    if len(d) >= 3:                                   # NumPy linear trend line
        slope, intercept = np.polyfit(d["attempt"], d["percent"], 1)
        ax.plot(d["attempt"], slope * d["attempt"] + intercept, "--", color=ACCENT, label="Trend")
        ax.legend()
    ax.set(ylim=(0, 105), xlabel="Exam attempt", ylabel="Score %", title="Score trend (Seaborn)")
    ax.set_xticks(d["attempt"])
    fig.tight_layout()
    return fig


def chart_by_difficulty(df):
    avg = (df.groupby("difficulty")["percent"].mean()
           .reindex(["Easy", "Medium", "Hard", "Mixed"]).dropna().reset_index())
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(avg["difficulty"], avg["percent"], color=["#22c55e", "#f59e0b", "#ef4444", SECONDARY][:len(avg)])
    for rect, v in zip(bars, avg["percent"]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set(ylim=(0, 110), ylabel="Average score %", title="Average score by difficulty (Matplotlib)")
    fig.tight_layout()
    return fig


def chart_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sns.histplot(df["percent"], bins=np.arange(0, 110, 10), color=SECONDARY, ax=ax)
    ax.set(xlabel="Score %", ylabel="Exams", title="Score distribution (Seaborn)")
    fig.tight_layout()
    return fig


def chart_topics(df):
    avg = df.groupby("topic")["percent"].mean().sort_values().tail(8)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = [("#22c55e" if v >= 70 else "#f59e0b" if v >= 50 else "#ef4444") for v in avg.values]
    ax.barh(avg.index, avg.values, color=colors)
    ax.set(xlim=(0, 100), xlabel="Average score %", title="Performance by topic (Matplotlib)")
    fig.tight_layout()
    return fig


def chart_heatmap(activity):
    d = pd.DataFrame(activity)
    days = pd.date_range(end=pd.Timestamp.today().normalize(), periods=14)
    d = d[d["date"].isin(days.strftime("%Y-%m-%d"))]
    if d.empty:
        return None
    pivot = d.groupby(["type", "date"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=days.strftime("%Y-%m-%d"), fill_value=0)
    pivot.columns = [c[5:] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Purples", cbar=False, linewidths=.5, ax=ax)
    ax.set(title="Learning activity - last 14 days (Seaborn)", xlabel="", ylabel="")
    fig.tight_layout()
    return fig


def chart_progress_donut(done, total):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    pct = 100 * done / total if total else 0
    ax.pie([done, max(total - done, 0)] if total else [0, 1], colors=[SECONDARY, "#e5e7eb"],
           startangle=90, counterclock=False, wedgeprops=dict(width=0.32))
    ax.text(0, 0, f"{pct:.0f}%", ha="center", va="center", fontsize=24, fontweight="bold", color=PRIMARY)
    ax.set_title("Roadmap progress (Matplotlib)")
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
    html = (FIREWORKS_HTML.replace("__TITLE__", title).replace("__SCORE__", f"{score}/{total} ({pct:.0f}%)")
            .replace("__MESSAGE__", msg))
    components.html(html, height=440)
    if pct >= 70:
        st.balloons()


# =============================================================================
# 7. PAGES  (Streamlit)
# =============================================================================
def run_ai(label, func):
    """Run an AI task with a spinner and uniform error display. Returns None on failure."""
    try:
        with st.spinner(label):
            return func()
    except AIError as e:
        st.error(str(e))
        return None


def page_home():
    hero("🌧️ Arora Rain", "Your personal AI learning companion - learn, summarise, practise and grow.")
    data, df = load_data(), exams_df()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topics completed", len(data["completed_topics"]))
    c2.metric("Exams taken", len(df))
    c3.metric("Average score", f"{np.mean(df['percent']):.0f}%" if not df.empty else "-")
    c4.metric("Day streak 🔥", current_streak())
    st.write("")
    feats = [("🧠 AI Learning Assistant", "Ask about any topic and get a clear explanation with examples."),
             ("📝 Lecture Summarizer", "Short and detailed summaries with key points."),
             ("📒 Smart Notes", "Organised notes with headings, definitions and examples."),
             ("🎯 AI Exam Generator", "10-100 MCQs, difficulty levels, marks and explanations."),
             ("🗺️ Learning Roadmap", "A personalised plan with milestones and resources."),
             ("📊 Progress Dashboard", "Track scores, topics and activity with live charts.")]
    cols = st.columns(3)
    for i, (t, d) in enumerate(feats):
        with cols[i % 3]:
            card(t, f"<p style='margin:0;opacity:.85'>{d}</p>")
    if not get_api_key():
        st.warning("🔑 Set up your free Gemini API key to enable the AI features:")
        st.markdown(SETUP_HELP)


def page_assistant():
    hero("🧠 AI Learning Assistant", "Enter any topic and get an easy explanation with examples.")
    if not require_ai():
        return
    c1, c2 = st.columns([3, 1])
    topic = c1.text_input("What do you want to learn?", placeholder="e.g. Photosynthesis, Recursion, Inflation")
    level = c2.selectbox("Explain for", ["Beginner", "Intermediate", "Advanced"])
    if st.button("✨ Explain it"):
        topic = topic.strip()
        if len(topic) < 2:
            st.error("Please enter a topic (at least 2 characters).")
        else:
            text = run_ai("Thinking...", lambda: ai_explain(topic, level))
            if text:
                st.session_state["explanation"] = {"topic": topic, "text": text}
                log_activity("Explain", topic)
    exp = st.session_state.get("explanation")
    if exp:
        st.markdown(f"### 📖 {exp['topic']}")
        st.markdown(exp["text"])
        if st.button("✅ Mark topic as studied"):
            mark_completed(exp["topic"], "assistant")
            st.success("Added to your completed topics!")
        st.download_button("⬇️ Download explanation (.md)", f"# {exp['topic']}\n\n{exp['text']}",
                           f"{safe_name(exp['topic'])}.md", "text/markdown")


def get_source_text(key):
    """Text area + .txt/.md upload used by the summarizer and notes pages."""
    text = st.text_area("Paste lecture text", height=220, key=f"{key}_text",
                        placeholder="Paste your lecture notes or transcript here...")
    up = st.file_uploader("...or upload a text file (.txt or .md)", type=["txt", "md"], key=f"{key}_up")
    if up is not None:
        try:
            extracted = up.read().decode("utf-8", errors="ignore")
            if extracted.strip():
                st.caption(f"Loaded {len(extracted.split())} words from {up.name}")
                text = extracted
        except Exception as e:
            st.error(f"Could not read '{up.name}': {e}")
    return text.strip()


def page_summarizer():
    hero("📝 Lecture Summarizer", "Turn long lectures into short and detailed summaries.")
    if not require_ai():
        return
    text = get_source_text("sum")
    if st.button("Summarise"):
        if len(text.split()) < 40:
            st.error("Please provide at least 40 words of text.")
        else:
            res = run_ai("Summarising...", lambda: ai_summary(text))
            if res:
                st.session_state["summary"] = res
                log_activity("Summarize", f"{len(text.split())} words")
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
    hero("📒 Smart Notes Generator", "Convert lecture content or a topic into organised notes.")
    if not require_ai():
        return
    title = st.text_input("Notes title (optional)")
    mode = st.radio("Source", ["Paste / upload lecture", "Generate from a topic"], horizontal=True)
    text = topic = ""
    if mode == "Paste / upload lecture":
        text = get_source_text("notes")
    else:
        topic = st.text_input("Topic", placeholder="e.g. Machine Learning")
    if st.button("Create notes"):
        from_topic = mode != "Paste / upload lecture"
        source = topic.strip() if from_topic else text
        if from_topic and len(source) < 2:
            st.error("Please enter a topic.")
        elif not from_topic and len(source.split()) < 40:
            st.error("Please provide at least 40 words of text.")
        else:
            notes = run_ai("Writing notes...", lambda: ai_notes(source, title.strip() or (source if from_topic else ""), from_topic))
            if notes:
                st.session_state["notes"] = notes
                log_activity("Notes", notes["title"])
    n = st.session_state.get("notes")
    if n:
        st.markdown(f"## {n['title']}")
        for s in n["sections"]:
            st.markdown(f"### {s.get('heading', '')}")
            for p in s.get("points", []):
                st.markdown(f"- {p}")
        if n["definitions"]:
            st.markdown("### 📚 Definitions")
            for d in n["definitions"]:
                st.markdown(f"- **{d.get('term', '')}** - {d.get('meaning', '')}")
        if n["examples"]:
            st.markdown("### 💡 Examples")
            for e in n["examples"]:
                st.markdown(f"- {e}")
        if n["key_points"]:
            st.markdown("### ⭐ Key points")
            for k in n["key_points"]:
                st.markdown(f"- {k}")
        download_buttons(notes_blocks(n), safe_name(n["title"]), "notes")


def page_exam():
    hero("🎯 AI Exam Generator", "Practise with custom MCQs, get marks and explanations.")
    if not require_ai():
        return
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
            bar = st.progress(0.0, text=f"Creating {n} questions...")
            try:
                qs = ai_exam(topic.strip(), difficulty, n, progress=lambda p: bar.progress(p))
            except AIError as e:
                bar.empty()
                st.error(str(e))
                return
            bar.empty()
            st.session_state["exam"] = {"id": uuid.uuid4().hex[:8], "topic": topic.strip(),
                                        "difficulty": difficulty, "questions": qs, "submitted": False}
            st.rerun()
        return

    qs = ex["questions"]
    st.markdown(f"### {ex['topic']}  <span class='pill'>{ex['difficulty']}</span>"
                f"<span class='pill'>{len(qs)} questions</span>", unsafe_allow_html=True)

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
        st.markdown("### 🔍 Review of incorrect answers")
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
        download_buttons(exam_blocks(ex["topic"], ex["difficulty"], score, total, qs, ex["answers"]),
                         "exam_report", "exam")
        if st.button("🔄 Start a new exam"):
            st.session_state["exam"] = None
            st.rerun()


def toggle_topic(topic, key):
    if st.session_state.get(key):
        mark_completed(topic, "roadmap")
    else:
        unmark_completed(topic)


def resource_link(title, rtype, field):
    q = quote_plus(f"{title} {field}")
    if str(rtype).lower() == "video":
        return f"https://www.youtube.com/results?search_query={q}"
    return f"https://www.google.com/search?q={q}"


def page_roadmap():
    hero("🗺️ Personalised Learning Roadmap", "Tell us your goal - Gemini plans the journey.")
    if not require_ai():
        return
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
            rm = run_ai("Designing your roadmap...", lambda: ai_roadmap(field.strip(), level, hours))
            if rm:
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
    st.caption(f"{rm.get('summary', '')}  •  about {rm['total_weeks']} weeks")
    done_set = {t["topic"].lower() for t in load_data()["completed_topics"]}
    for i, p in enumerate(rm["phases"]):
        with st.expander(f"Phase {i + 1}: {p['name']}  -  {p['weeks']} week(s)", expanded=(i == 0)):
            for j, t in enumerate(p["topics"]):
                key = f"rm_{i}_{j}"
                st.checkbox(t, value=t.lower() in done_set, key=key, on_change=toggle_topic, args=(t, key))
            st.markdown(f"🏁 **Milestone:** {p['milestone']}")
            for r in p["resources"]:
                link = resource_link(r.get("title", ""), r.get("type", ""), rm.get("field", ""))
                st.markdown(f"- [{r.get('title', '')}]({link})  <span class='pill'>{r.get('type', '')}</span>",
                            unsafe_allow_html=True)
    download_buttons(roadmap_blocks(rm), "learning_roadmap", "rm")


def page_dashboard():
    hero("📊 Progress Dashboard", "See how far you've come.")
    data, df = load_data(), exams_df()
    pct = df["percent"].to_numpy(dtype=float)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Topics completed", len(data["completed_topics"]))
    c2.metric("Exams taken", len(df))
    c3.metric("Average score", f"{np.mean(pct):.0f}%" if len(pct) else "-")
    c4.metric("Best score", f"{np.max(pct):.0f}%" if len(pct) else "-")
    if len(pct) >= 2:
        slope = np.polyfit(np.arange(len(pct)), pct, 1)[0]
        c5.metric("Trend", f"{slope:+.1f}% / exam")
    else:
        c5.metric("Day streak 🔥", current_streak())
    if len(pct) >= 2:
        st.caption(f"Consistency (std. deviation): {np.std(pct):.1f} points  •  "
                   f"Median: {np.median(pct):.0f}%  •  Day streak: {current_streak()} 🔥")

    rm = data.get("roadmap")
    left, right = st.columns(2)
    with left:
        if len(df):
            show_fig(chart_score_trend(df))
        else:
            st.info("Take an exam to see your score trend.")
    with right:
        if rm:
            topics = [t for p in rm["phases"] for t in p["topics"]]
            done = {t["topic"].lower() for t in data["completed_topics"]}
            show_fig(chart_progress_donut(sum(t.lower() in done for t in topics), len(topics)))
        else:
            st.info("Create a roadmap to see your progress donut.")
    if len(df):
        a, b = st.columns(2)
        with a:
            show_fig(chart_by_difficulty(df))
        with b:
            show_fig(chart_distribution(df))
        show_fig(chart_topics(df))
    if data["activity"]:
        fig = chart_heatmap(data["activity"])
        if fig:
            show_fig(fig)

    st.markdown("### ✅ Completed topics")
    if data["completed_topics"]:
        st.dataframe(pd.DataFrame(data["completed_topics"]), use_container_width=True, hide_index=True)
    else:
        st.caption("Nothing yet - study a topic and mark it as completed!")
    if len(df):
        st.markdown("### 📝 Exam history")
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### 🎁 What should I learn next?")
    if get_api_key():
        if st.button("Ask Gemini for recommendations"):
            recs = run_ai("Thinking...", lambda: ai_recommendations(df, [t["topic"] for t in data["completed_topics"]]))
            if recs:
                st.session_state["recs"] = recs
        for r in st.session_state.get("recs", []):
            st.markdown(f"- **{r['topic']}** - {r.get('reason', '')}")
    else:
        st.caption("Add your Gemini key to get personalised recommendations.")
    with st.expander("⚠️ Danger zone"):
        if st.button("Reset all progress"):
            save_data(default_data())
            st.success("Progress cleared.")
            st.rerun()


# =============================================================================
# 8. MAIN: sidebar navigation
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
        st.markdown("**✨ Gemini AI**")
        st.text_input("Model", value=get_model(), key="model",
                      help="Any Gemini model name from aistudio.google.com")
        if get_api_key():
            st.caption("🔑 API key loaded (secrets / env)")
            if st.button("🔌 Test connection"):
                res = run_ai("Testing...", lambda: generate("Reply with the single word: OK", temperature=0))
                if res:
                    st.success("Gemini is connected!")
        else:
            st.caption("⚠️ No API key found - see Home page")
    try:
        PAGES[choice]()
    except Exception as e:                            # last-resort safety net
        st.error(f"Something went wrong: {e}")


if __name__ == "__main__":
    main()
