"""
app.py
------
Flask web application for the Resume Screening System.

Two screening modes, exposed both as JSON APIs (for the frontend JS) and
usable directly via curl/Postman:

  POST /api/predict
      body: {"resume_text": "..."}
      -> predicted experience level + confidence + extracted skills

  POST /api/rank
      body: {"jd_text": "...", "required_skills": ["Python", "SQL", ...]}
      -> ranks every resume in the dataset against this JD, best match first

Run locally:
    pip install -r requirements.txt
    python src/train_classifier.py   # one-time: trains & saves the model
    python app.py
    -> open http://127.0.0.1:5000
"""

import os
import sys
import pandas as pd
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import screen_resume
from screening_engine import rank_resumes
from preprocess import split_skills

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "resumes_dataset.csv")

app = Flask(__name__)

_dataset_cache = None


def get_dataset_resumes():
    """Loads the dataset once and formats it as the list-of-dict shape rank_resumes() expects."""
    global _dataset_cache
    if _dataset_cache is None:
        df = pd.read_csv(DATA_PATH)
        resumes = []
        for _, row in df.iterrows():
            resumes.append({
                "id": row["filename"],
                "name": row["name"],
                "experience_level": row["experience_level"],
                "resume_text": row["resume_text"],
                "skills": split_skills(row["skills"]),
            })
        _dataset_cache = resumes
    return _dataset_cache


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) or {}
    resume_text = (data.get("resume_text") or "").strip()

    if not resume_text:
        return jsonify({"error": "resume_text is required"}), 400

    try:
        result = screen_resume(resume_text)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


@app.route("/api/rank", methods=["POST"])
def api_rank():
    data = request.get_json(force=True) or {}
    jd_text = (data.get("jd_text") or "").strip()
    required_skills = data.get("required_skills") or []

    if not jd_text:
        return jsonify({"error": "jd_text is required"}), 400
    if isinstance(required_skills, str):
        required_skills = [s.strip() for s in required_skills.split(",") if s.strip()]

    resumes = get_dataset_resumes()
    ranked = rank_resumes(jd_text, required_skills, resumes)

    # Trim the heavy resume_text out of the response payload (keep it light for the UI)
    for r in ranked:
        r.pop("resume_text", None)
        r.pop("skills", None)

    top_n = int(data.get("top_n", 10))
    return jsonify({"ranked_candidates": ranked[:top_n], "total_screened": len(ranked)})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})



