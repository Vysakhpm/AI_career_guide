import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define career clusters
career_clusters = {
    "STEM": ["technology", "coding", "math", "logic", "engineering", "science"],
    "Arts & Creative Industries": ["writing", "music", "drawing", "creativity", "acting"],
    "Sports & Physical Performance": ["sports", "fitness", "training", "teamwork", "discipline"],
    "Business & Entrepreneurship": ["marketing", "sales", "leadership", "management", "strategy"],
    "Healthcare & Social Services": ["empathy", "helping others", "listening", "support", "caregiving"],
}

# Explanation templates
def explain_path(career):
    templates = {
        "STEM": "With your interest in logical thinking and problem-solving, STEM offers paths like engineering, programming, or data science.",
        "Arts & Creative Industries": "Your creativity and expression can shine in careers like writing, filmmaking, or design.",
        "Sports & Physical Performance": "You seem driven and active—sports and physical careers offer discipline, teamwork, and passion.",
        "Business & Entrepreneurship": "You show leadership and strategy—perfect for roles in management, marketing, or launching a startup.",
        "Healthcare & Social Services": "You care about others deeply—consider roles in nursing, counseling, or social work.",
    }
    return templates.get(career, "This is a promising field!")

# Semantic mapping
def map_to_career(user_text):
    user_embed = model.encode(user_text, convert_to_tensor=True)
    scores = {}
    for cluster, keywords in career_clusters.items():
        keyword_embed = model.encode(keywords, convert_to_tensor=True)
        score = util.cos_sim(user_embed, keyword_embed).mean().item()
        scores[cluster] = round(score, 3)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:3]

# Streamlit App Layout
st.set_page_config(page_title="Career Path Recommender", layout="centered")

st.title("🎯 AI Career Path Recommender")
st.markdown("Tell me about your **interests, hobbies, and what excites you**. I’ll suggest a few career paths you might enjoy!")

with st.form("career_form"):
    name = st.text_input("👤 What's your name?", "")
    hobby = st.text_area("🎨 What hobbies or activities do you enjoy?")
    subjects = st.text_area("📚 What subjects or topics excite you?")
    proud = st.text_area("🏆 Describe something you did that made you proud")
    deep_interest = st.text_area("🌍 What kind of problems in the world do you care about solving?")
    submitted = st.form_submit_button("Get My Career Paths 🚀")

if submitted:
    if not any([hobby, subjects, proud, deep_interest]):
        st.warning("Please tell me something about your interests.")
    else:
        user_input = " ".join([hobby, subjects, proud, deep_interest])
        top_paths = map_to_career(user_input)

        st.subheader(f"🔍 Hi {name}, here are your top recommended career clusters:")

        for path, score in top_paths:
            st.markdown(f"### 🔹 {path} (score: {score})")
            st.write(explain_path(path))
