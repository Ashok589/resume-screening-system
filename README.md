# Resume Screening System

An NLP-based resume screening system: classifies a resume's experience level
(Fresher / Mid-Level) and ranks a batch of resumes against a job description
using skill overlap + text similarity. Built end-to-end — data → preprocessing
→ model training → evaluation → Flask web app → deployment.

```
resume_screening_system/
├── data/
│   └── resumes_dataset.csv        # your dataset
├── notebooks/
│   └── resume_screening_notebook.ipynb   # exploratory notebook (same structure as your fake-news .ipynb)
├── src/
│   ├── preprocess.py               # text cleaning, no NLTK download needed
│   ├── eda.py                      # step 1-2: load, clean, EDA plots
│   ├── train_classifier.py         # step 3-10: train, cross-validate, save model
│   ├── screening_engine.py         # skill extraction + JD matching/ranking
│   └── predict.py                  # inference layer used by the web app
├── models/                          # saved model artifacts (created by training)
├── templates/index.html             # web UI
├── static/{style.css, app.js}       # web UI
├── app.py                           # Flask app (routes + JSON API)
├── requirements.txt
└── Procfile                         # for Render/Railway/Heroku-style deploys
```

`notebooks/resume_screening_notebook.ipynb` mirrors your fake-news notebook's
section structure (load & clean → preprocess → features → split → model
comparison → train → evaluate → save → test on real data) plus a bonus
JD-matching section, all with real executed outputs and charts already
embedded — open it directly in Jupyter or VS Code, no need to rerun it
unless you want to. `src/*.py` is the same logic refactored into the
production Flask app.

## Why this architecture (and not a BiLSTM like the fake-news notebook)

Your fake-news notebook trains a Word2Vec embedding + Bidirectional LSTM.
That works there because the dataset is large. **This resume dataset has
only 100 rows.** A neural network with learned embeddings needs thousands of
examples to generalize — with 100 rows it would just memorize the training
set. So this project uses **TF-IDF + classical ML** (Logistic Regression /
Linear SVM / Random Forest, picked by cross-validated F1) instead. This is
the standard, defensible choice for small text datasets, and it's also
faster to train and deploy.

One more honest note: cross-validation on this dataset comes back at a
perfect 1.0 F1 score. That's a signal the dataset is fairly clean/synthetic
with very separable language patterns (e.g. "recent graduate" vs explicit
years of experience phrasing) — not a guarantee the model will perform that
well on messier, real-world resumes. If you plan to use this on real
resumes, treat this as a working baseline, not a finished product.

There's also a minor data quirk worth knowing about: the `skills` column in
the source CSV contains a few fragment entries (`"Deep"`, `"MS"`,
`"Computer"`, `"Machine"`, `"Object"`, `"Generative"`) alongside the full
skill names they came from (`"Deep Learning"`, `"MS Excel"`, etc.). The skill
extractor in `screening_engine.py` prefers the longest match to avoid most
double-counting, but a fragment can still surface if it genuinely occurs
elsewhere in the text (e.g. "B.Sc in Computer Science" legitimately contains
the word "Computer"). Worth cleaning that column if you extend this dataset.

## What the system actually does

1. **Experience-level classifier** (`train_classifier.py` → `predict.py`)
   Cleans resume text → TF-IDF vectorizes it → predicts Fresher / Mid-Level
   with a confidence score.

2. **JD-based screening & ranking** (`screening_engine.py`)
   This is the actual "screening" behavior recruiters want: give it a job
   description and a list of required skills, and it scores every resume
   with:
   - `skill_match_pct` — % of required skills the resume actually has
   - `text_similarity_pct` — TF-IDF cosine similarity to the JD overall
   - `final_score` = 60% skill match + 40% text similarity, then ranks
     candidates best-first with matched/missing skills called out.

## Run it locally

```bash
cd resume_screening_system
pip install -r requirements.txt

# 1. (optional) EDA — produces static/class_balance.png, top_skills.png, etc.
python src/eda.py

# 2. Train & save the model (required before running the app)
python src/train_classifier.py

# 3. Start the web app
python app.py
```

Then open **http://127.0.0.1:5000**. Two tabs:
- **Screen one resume** — paste resume text, get predicted level + skills.
- **Rank against a job** — paste a JD + required skills, get the dataset
  ranked best-match-first.

### API (used by the frontend, also usable directly)

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "3 years experience in Python, SQL and Docker..."}'

curl -X POST http://127.0.0.1:5000/api/rank \
  -H "Content-Type: application/json" \
  -d '{"jd_text": "Hiring a Python backend dev", "required_skills": "Python, SQL, Git"}'
```

## Deployment

The app is a standard Flask app, so it deploys anywhere that runs Python.
Two easy free/low-cost options:

### Option A — Render.com (recommended, simplest)
1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com) → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt && python src/train_classifier.py`
4. Start command: `gunicorn app:app`
5. Deploy. Render gives you a public URL.

### Option B — Railway.app
1. Push to GitHub, then "New Project" → "Deploy from GitHub repo" on
   [railway.app](https://railway.app).
2. Railway auto-detects Python; set the start command to `gunicorn app:app`
   and add a pre-deploy/build step running `python src/train_classifier.py`
   (or just commit the `models/` folder to the repo so training isn't
   needed at deploy time — it's small).

### Option C — Any VPS / your own server
```bash
pip install -r requirements.txt
python src/train_classifier.py
gunicorn --bind 0.0.0.0:8000 app:app
```
Put nginx in front of it for TLS if it's public-facing.

A `Procfile` is included for Render/Railway/Heroku-style platforms that
read one automatically.

**Important:** commit the `models/*.joblib` files (or run the training
script as part of your build step) — the app raises a clear error on
`/api/predict` if it can't find them.

## Extending this for real use

- Swap in real job-category labels (e.g. "Data Scientist", "HR", "Java
  Developer") if you get a richer dataset — that turns this into a
  multi-class classifier, which is closer to what most public "resume
  screening" datasets (like the Kaggle Resume Dataset) are built around.
- Add PDF/DOCX upload + text extraction (`pdfplumber` / `python-docx`) so
  recruiters can upload files directly instead of pasting text.
- Replace TF-IDF with sentence embeddings (e.g. `sentence-transformers`) for
  the JD-matching step once you have enough data/compute budget — better
  semantic matching than pure keyword/n-gram overlap.
