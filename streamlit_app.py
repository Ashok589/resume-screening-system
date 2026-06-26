"""
streamlit_app.py
-----------------
Streamlit version of the Resume Screening System (replaces app.py / Flask
for deployment on Streamlit Community Cloud or `streamlit run`).

Reuses the same underlying logic as the Flask app — src/preprocess.py,
src/predict.py, src/screening_engine.py — just a different UI layer on top.

Run locally:
    pip install -r requirements.txt
    python src/train_classifier.py     # one-time: trains & saves the model
    streamlit run streamlit_app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import screen_resume
from screening_engine import rank_resumes
from preprocess import split_skills

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "resumes_dataset.csv")

SAMPLE_RESUME = """Ravi Mehta
ravi.mehta@email.com | +91 9876543210 | Bangalore, India

PROFESSIONAL SUMMARY
Software engineer with 3 years of experience building backend services in Python.
Comfortable working across the stack from PostgreSQL schema design to deployment.

SKILLS
Python, SQL, PostgreSQL, Git, Docker, System Design, Data Structures

EXPERIENCE
Backend Engineer - Acme Analytics (2 years)
- Built REST APIs serving 50k+ daily requests using Python and PostgreSQL
- Set up CI/CD pipelines with Docker and Git-based workflows
- Mentored 2 junior engineers on system design fundamentals

EDUCATION
B.Tech in Computer Science"""

SAMPLE_JD = """We are hiring a Backend Engineer comfortable with Python and SQL.
Experience with PostgreSQL, Docker and Git is required. Familiarity with
system design concepts is a strong plus."""

SAMPLE_SKILLS = "Python, SQL, PostgreSQL, Docker, Git, System Design"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="🗂️",
    layout="wide",
)

# Small bit of custom styling so skills render as pill-shaped chips
st.markdown(
    """
    <style>
    .chip {
        display: inline-block;
        font-family: monospace;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 999px;
        margin: 2px 4px 2px 0;
    }
    .chip-matched { background: #E6F2EA; color: #2F7D52; border: 1px solid #BFDFC9; }
    .chip-missing { background: #FBEAE7; color: #C0392B; border: 1px solid #F0CCC4; }
    .chip-neutral { background: #F0ECE0; color: #2B3138; border: 1px solid #E1DACB; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading (cached so the CSV isn't re-read on every interaction)
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset_resumes():
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
    return resumes


def chips_html(items, css_class):
    if not items:
        return ""
    return "".join(f'<span class="chip {css_class}">{s}</span>' for s in items)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🗂️ Resume Screening System")
st.caption("TF-IDF classifier + job-description match scoring")

tab1, tab2 = st.tabs(["📄  Screen one resume", "📊  Rank against a job"])


# ---------------------------------------------------------------------------
# TAB 1 — single resume
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Predict experience level & extract skills")
    st.write(
        "Paste raw resume text below. The model classifies the candidate as "
        "**Fresher** or **Mid-Level** and pulls out the skills it recognizes."
    )

    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""

    def _load_sample_resume():
        st.session_state.resume_text = SAMPLE_RESUME

    col1, col2 = st.columns(2)

    with col1:
        st.text_area("Resume text", key="resume_text", height=380,
                      placeholder="Paste the full resume text here...")
        c1, c2 = st.columns(2)
        with c1:
            st.button("Load a sample resume", on_click=_load_sample_resume,
                       use_container_width=True)
        with c2:
            predict_clicked = st.button("Screen this resume", type="primary",
                                         use_container_width=True)

    with col2:
        st.markdown("**Result**")
        result_box = st.container(border=True)

        if predict_clicked:
            if not st.session_state.resume_text.strip():
                result_box.warning("Paste some resume text first.")
            else:
                try:
                    with st.spinner("Screening resume..."):
                        result = screen_resume(st.session_state.resume_text)
                except FileNotFoundError as e:
                    result_box.error(str(e))
                else:
                    label = result["predicted_label"]
                    confidence = result["confidence"]
                    skills = result["extracted_skills"]

                    badge = "🟦 Fresher" if label == "Fresher" else "🟧 Mid-Level"
                    result_box.markdown(f"### {badge}")

                    if confidence is not None:
                        result_box.write(f"**Confidence:** {confidence}%")
                        result_box.progress(min(confidence / 100, 1.0))
                    else:
                        result_box.caption("Confidence not available for this model.")

                    result_box.caption(f"Model used: `{result['model_used']}`")

                    result_box.markdown(f"**Skills detected ({len(skills)})**")
                    if skills:
                        result_box.markdown(chips_html(skills, "chip-neutral"),
                                             unsafe_allow_html=True)
                    else:
                        result_box.caption("No known skills detected.")
        else:
            result_box.caption("Results will appear here once you screen a resume.")


# ---------------------------------------------------------------------------
# TAB 2 — batch ranking against a job description
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Rank the dataset against a job description")
    st.write(
        "Describe the role and list the must-have skills. Every resume in "
        "the dataset is scored — 60% skill overlap, 40% text relevance — "
        "and ranked best-match first."
    )

    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""
    if "skills_text" not in st.session_state:
        st.session_state.skills_text = ""

    def _load_sample_jd():
        st.session_state.jd_text = SAMPLE_JD
        st.session_state.skills_text = SAMPLE_SKILLS

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.text_area("Job description", key="jd_text", height=180,
                      placeholder="e.g. We are hiring a Python backend developer...")
        st.text_input("Required skills (comma-separated)", key="skills_text",
                       placeholder="Python, SQL, Git, PostgreSQL")

        c1, c2 = st.columns(2)
        with c1:
            st.button("Load a sample job", on_click=_load_sample_jd,
                       use_container_width=True)
        with c2:
            rank_clicked = st.button("Rank candidates", type="primary",
                                      use_container_width=True)

        top_n = st.slider("Show top N candidates", min_value=3, max_value=20, value=10)

    with col2:
        st.markdown("**Ranked candidates**")

        if rank_clicked:
            if not st.session_state.jd_text.strip():
                st.warning("Paste a job description first.")
            else:
                required_skills = [s.strip() for s in st.session_state.skills_text.split(",") if s.strip()]
                resumes = load_dataset_resumes()

                with st.spinner("Ranking candidates..."):
                    ranked = rank_resumes(st.session_state.jd_text, required_skills, resumes)

                st.caption(f"Screened {len(ranked)} resumes — showing top {min(top_n, len(ranked))}")

                for i, c in enumerate(ranked[:top_n], start=1):
                    with st.container(border=True):
                        top_col, score_col = st.columns([3, 1])
                        top_col.markdown(f"**#{i}  {c['name']}**  ·  _{c['experience_level']}_")
                        score_col.markdown(f"**{c['final_score']}% match**")

                        st.progress(min(c["final_score"] / 100, 1.0))

                        meta_col1, meta_col2 = st.columns(2)
                        meta_col1.caption(f"Skill overlap: {c['skill_match_pct']}%")
                        meta_col2.caption(f"Text relevance: {c['text_similarity_pct']}%")

                        if c["matched_skills"] or c["missing_skills"]:
                            st.markdown(
                                chips_html(c["matched_skills"], "chip-matched")
                                + chips_html(c["missing_skills"], "chip-missing"),
                                unsafe_allow_html=True,
                            )
        else:
            st.caption("Ranked candidates will appear here.")


st.divider()
st.caption(
    "Built with TF-IDF + Logistic Regression for classification, and "
    "TF-IDF cosine similarity + skill overlap for JD matching."
)
