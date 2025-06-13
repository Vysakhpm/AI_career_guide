# career_guide_ai.py

from sentence_transformers import SentenceTransformer, util

# Career clusters with sample keywords
career_clusters = {
    "STEM": ["technology", "coding", "math", "logic", "engineering", "science"],
    "Arts & Creative Industries": ["writing", "music", "drawing", "creativity", "acting"],
    "Sports & Physical Performance": ["sports", "fitness", "training", "teamwork", "discipline"],
    "Business & Entrepreneurship": ["marketing", "sales", "leadership", "management", "strategy"],
    "Healthcare & Social Services": ["empathy", "helping others", "listening", "support", "caregiving"],
}

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def ask_user_questions():
    print("👋 Hi! I'm your AI career guide.")
    name = input("What’s your name? ")
    print(f"Nice to meet you, {name}! Let’s explore your interests.\n")

    answers = []
    questions = [
        "1. What hobbies or activities do you enjoy the most?",
        "2. What school subjects or topics excite you?",
        "3. Is there anything you do that makes you lose track of time?",
        "4. Describe a time you felt proud of something you made or did."
    ]

    for q in questions:
        ans = input(q + "\n> ")
        answers.append(ans)

    full_text = " ".join(answers)
    if len(full_text.split()) < 30:
        print("\nHmm, I’d love to learn a bit more. Can you answer this too?")
        extra = input("👉 What kind of problems in the world do you care about solving?\n> ")
        full_text += " " + extra

    return full_text

def map_to_career(user_text):
    user_embed = model.encode(user_text, convert_to_tensor=True)
    scores = {}

    for cluster, keywords in career_clusters.items():
        keyword_embed = model.encode(keywords, convert_to_tensor=True)
        score = util.cos_sim(user_embed, keyword_embed).mean().item()
        scores[cluster] = round(score, 3)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:3]

def explain_path(career):
    templates = {
        "STEM": "With your interest in logical thinking and problem-solving, STEM offers paths like engineering, programming, or data science.",
        "Arts & Creative Industries": "Your creativity and expression can shine in careers like writing, filmmaking, or design.",
        "Sports & Physical Performance": "You seem driven and active—sports and physical careers offer discipline, teamwork, and passion.",
        "Business & Entrepreneurship": "You show leadership and strategy—perfect for roles in management, marketing, or launching a startup.",
        "Healthcare & Social Services": "You care about others deeply—consider roles in nursing, counseling, or social work.",
    }
    return templates.get(career, "This is a promising field!")

def run_career_guide():
    user_text = ask_user_questions()
    top_paths = map_to_career(user_text)

    print("\n📊 Based on your responses, here are some career paths you might enjoy:\n")
    for path, score in top_paths:
        print(f"🔹 {path} (match score: {score})")
        print(f"💡 {explain_path(path)}\n")

if __name__ == "__main__":
    run_career_guide()
