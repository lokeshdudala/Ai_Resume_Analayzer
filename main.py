from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
import shutil
import json
import numpy as np
import torch
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ---------------- Deterministic Setup ----------------

torch.manual_seed(42)
np.random.seed(42)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load AI Model ----------------

print("Loading AI model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")

# ---------------- Load Skills ----------------

with open("skills.json", "r") as f:
    skill_categories = json.load(f)

skills_list = []
for category in skill_categories.values():
    skills_list.extend(category)

skills_list = sorted(list(set(skills_list)))

# ---------------- Role → Skill Mapping ----------------

role_skill_map = {
    "full stack developer": [
        "react", "node", "express", "mongodb",
        "html", "css", "javascript", "rest api"
    ],
    "backend developer": [
        "python", "java", "node", "sql",
        "mongodb", "rest api"
    ],
    "frontend developer": [
        "react", "html", "css", "javascript"
    ],
    "database architect": [
        "sql", "mongodb", "database design",
        "indexing", "schema design"
    ],
    "cloud engineer": [
        "aws", "docker"
    ],
    "data scientist": [
        "python", "machine learning", "sql"
    ]
}

# ---------------- PDF Extraction ----------------

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---------------- Skill Extraction (Word Boundary Safe) ----------------

def extract_skills(text, skills):
    text = text.lower()
    detected = []

    for skill in skills:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text):
            detected.append(skill)

    return sorted(list(set(detected)))

# ---------------- Fuzzy Role Expansion ----------------

def expand_role_skills(job_description):
    jd_lower = job_description.lower()

    for role, mapped_skills in role_skill_map.items():
        if role in jd_lower:
            return mapped_skills

    possible_roles = list(role_skill_map.keys())
    match = get_close_matches(jd_lower, possible_roles, n=1, cutoff=0.6)

    if match:
        return role_skill_map[match[0]]

    return []

# ---------------- Stable Semantic Similarity ----------------

def semantic_similarity(resume_text, job_description):
    resume_embedding = np.array(model.encode(resume_text)).reshape(1, -1)
    jd_embedding = np.array(model.encode(job_description)).reshape(1, -1)

    similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]

    # Round early to avoid floating fluctuation
    return round(float(similarity * 100), 4)

# ---------------- Stable Job Match Logic ----------------

def calculate_job_match(resume_text, job_description, resume_skills, jd_skills):
    semantic_score = semantic_similarity(resume_text, job_description)

    if jd_skills:
        skill_overlap = round(
            (len(set(resume_skills) & set(jd_skills)) / len(jd_skills)) * 100,
            4
        )
    else:
        skill_overlap = 0.0

    word_count = len(job_description.split())

    if word_count <= 4:
        final_score = (0.7 * skill_overlap) + (0.3 * semantic_score)
    else:
        final_score = (0.5 * skill_overlap) + (0.5 * semantic_score)

    return round(float(min(final_score, 100)), 2)

# ---------------- Stable ATS Score ----------------

def analyze_ats(match_score, resume_skills, jd_skills):
    ats_score = match_score * 0.6

    skill_alignment = len(set(resume_skills) & set(jd_skills))
    ats_score += skill_alignment * 6

    if len(resume_skills) >= 8:
        ats_score += 10

    return round(float(min(ats_score, 100)), 2)

# ---------------- Learning Suggestions ----------------

def generate_learning_suggestions(missing_skills):
    suggestions = {}
    for skill in missing_skills:
        suggestions[skill] = [
            {
                "title": f"{skill.title()} Fundamentals",
                "platform": "General Resource",
                "level": "Beginner"
            },
            {
                "title": f"Advanced {skill.title()}",
                "platform": "Online Platform",
                "level": "Advanced"
            }
        ]
    return suggestions

# ---------------- Interview Questions ----------------

def generate_interview_questions(skills):
    questions = {}
    for skill in skills:
        questions[skill] = [
            f"What is {skill} and where is it used?",
            f"Explain architecture and design principles of {skill}.",
            f"How would you optimize performance in {skill}?"
        ]
    return questions

# ---------------- Routes ----------------

@app.get("/")
def home():
    return {"message": "Stable AI Resume Intelligence Running 🚀"}

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resume_text = extract_text_from_pdf(file_location)

    resume_skills = extract_skills(resume_text, skills_list)
    jd_skills = extract_skills(job_description, skills_list)

    role_skills = expand_role_skills(job_description)
    jd_skills = sorted(list(set(jd_skills + role_skills)))

    match_score = calculate_job_match(
        resume_text,
        job_description,
        resume_skills,
        jd_skills
    )

    matched = sorted(list(set(resume_skills) & set(jd_skills)))
    missing = sorted(list(set(jd_skills) - set(resume_skills)))

    ats_score = analyze_ats(match_score, resume_skills, jd_skills)

    suggestions = generate_learning_suggestions(missing)

    question_base = matched + missing
    if not question_base:
        question_base = resume_skills

    questions = generate_interview_questions(sorted(question_base))

    return {
        "match_score_percent": match_score,
        "ats_score": ats_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "learning_suggestions": suggestions,
        "interview_questions": questions
    }
