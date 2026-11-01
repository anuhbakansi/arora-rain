# arora-rain
# 🌧️ Arora Rain - AI Learning Companion

Arora Rain is a Streamlit app powered by the **Google Gemini API** (official Google Gen AI Python SDK). It explains topics, summarises lectures, builds notes, generates exams, plans learning roadmaps and tracks your progress.

## Features

| Feature | What it does |
|---|---|
| AI Learning Assistant | Explains any topic in simple language with key concepts and examples |
| Lecture Summarizer | Short summary, detailed summary and key points (paste text or upload .txt/.md) |
| Smart Notes | Headings, definitions, examples and key points from a lecture or a topic |
| Exam Generator | 10-100 MCQs, Easy/Medium/Hard/Mixed, marks, explanations for wrong answers |
| Learning Roadmap | Field + skill level + weekly hours -> phases, topics, milestones, resources |
| Progress Dashboard | Scores, trends, topic performance, activity heatmap, roadmap progress |
| Celebration Screen | Fireworks animation and encouraging message after every exam |

Notes, summaries, roadmaps and exam reports can be downloaded as Markdown or plain text.

## Libraries and their roles

| Library | Role |
|---|---|
| **Streamlit** | Entire interface: sidebar navigation, forms, tabs, uploads, downloads, custom CSS gradients and animations |
| **Pandas** | Stores exams as DataFrames, groups scores by topic/difficulty, builds the activity pivot table |
| **NumPy** | Average/median/standard deviation, linear trend line (`polyfit`), array maths for charts |
| **Matplotlib** | Difficulty bar chart, topic performance chart, roadmap progress donut |
| **Seaborn** | Score trend line, score distribution histogram, activity heatmap |
| **google-genai** | Official Google Gen AI SDK - every AI feature calls Gemini through it |

Only the Python standard library is used besides these libraries.

## Installation

Requires Python 3.9 or newer.

```bash
# 1. (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
```

## Gemini API configuration

The API key is never written in the code and never typed into the web page. The app reads it from Streamlit secrets or an environment variable.

1. Open **https://aistudio.google.com**, sign in with your Google account and click **Get API key** (a free tier is available).
2. Provide the key in **one** of these ways:

   **Option A - Streamlit secrets (recommended).** Create the file `.streamlit/secrets.toml` next to `app.py`:

   ```toml
   GEMINI_API_KEY = "paste-your-key-here"
   ```

   **Option B - environment variable.**
   - macOS/Linux: `export GEMINI_API_KEY="paste-your-key-here"`
   - Windows (PowerShell): `$env:GEMINI_API_KEY="paste-your-key-here"`
   - Windows (cmd): `set GEMINI_API_KEY=paste-your-key-here`

3. Keep the key private: do not commit `.streamlit/secrets.toml` to Git (add it to `.gitignore`).

The default model is `gemini-2.5-flash`. You can change it in the sidebar, or set `GEMINI_MODEL` in secrets / environment.

## Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501. In the sidebar, press **Test connection** to confirm the key works.

## How to use

1. **Home** - overview and your key stats.
2. **AI Assistant** - enter a topic, read the explanation, mark it as studied.
3. **Summarizer / Smart Notes** - paste or upload a lecture and download the result.
4. **Exam Generator** - choose topic, difficulty and number of questions, answer, submit, enjoy the celebration and review mistakes. Scoring 60% or more marks the topic as completed.
5. **Roadmap** - build a plan and tick topics off as you finish them.
6. **Dashboard** - watch your progress, and ask Gemini what to learn next.

Progress is saved in `arora_data.json` in the folder where you run the app.

## Troubleshooting

- **"No Gemini API key found"** - check the spelling `GEMINI_API_KEY`, the location of `.streamlit/secrets.toml` (next to `app.py`, in the folder you run `streamlit` from) and restart the app.
- **"Google rejected the API key"** - create a new key in AI Studio and paste it again without quotes or spaces.
- **Rate limit (429)** - the free tier has per-minute limits. Wait a minute, or generate fewer questions at once.
- **"Model not found"** - set a current Gemini model name in the sidebar.
- **ModuleNotFoundError: google** - run `pip install -r requirements.txt` in the same environment you start Streamlit from.
