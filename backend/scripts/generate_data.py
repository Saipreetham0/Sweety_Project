import os
import random
import shutil

# Ensure directories exist
DATA_DIR = "backend/data"
HUMAN_DIR = os.path.join(DATA_DIR, "human")
AI_DIR = os.path.join(DATA_DIR, "ai")

os.makedirs(HUMAN_DIR, exist_ok=True)
os.makedirs(AI_DIR, exist_ok=True)

# Templates and Vocabulary
SKILLS = ["Python", "Java", "C++", "Project Management", "Data Analysis", "React", "AWS", "SQL"]
ROLES = ["Software Engineer", "Data Scientist", "Product Manager", "Analyst"]
UNIVERSITIES = ["State University", "City College", "Tech Institute", "Global University"]

def generate_human_resume(index):
    name = f"Human_Candidate_{index}"
    role = random.choice(ROLES)
    
    # Human traits: specific, sometimes choppy, direct, active verbs
    summary = f"I am a {role} with 5 years of experience. I worked on many projects using {random.choice(SKILLS)} and {random.choice(SKILLS)}. I want to solve hard problems and help the team."
    
    experience = f"""
    Experience:
    - Built a web app using {random.choice(SKILLS)}.
    - Fixed bugs in the production system.
    - Talked to clients about requirements.
    """
    
    text = f"Name: {name}\nRole: {role}\n\nSummary:\n{summary}\n{experience}\n\nEducation:\nBS in CS from {random.choice(UNIVERSITIES)}."
    
    with open(os.path.join(HUMAN_DIR, f"{name}.txt"), "w") as f:
        f.write(text)

def generate_ai_resume(index):
    name = f"AI_Candidate_{index}"
    role = random.choice(ROLES)
    
    # AI traits: buzzwords, "delve", "landscape", "testament", passive/flowery
    summary = f"As a meticulously crafted {role}, I strive to delve into the complex landscape of technology. My career is a testament to my ability to leverage the power of {random.choice(SKILLS)}."
    
    experience = f"""
    Professional Journey:
    - Orchestrated a tapestry of digital solutions utilizing {random.choice(SKILLS)}.
    - Navigated the complexities of backend architecture to foster a culture of innovation.
    - Poised to drive synergistic growth through data-driven paradigms.
    """
    
    text = f"Name: {name}\nRole: {role}\n\nProfessional Summary:\n{summary}\n{experience}\n\nAcademic Background:\nDegree obtained from {random.choice(UNIVERSITIES)}, underscoring a commitment to excellence."
    
    with open(os.path.join(AI_DIR, f"{name}.txt"), "w") as f:
        f.write(text)

def main():
    print(f"Generating data in {DATA_DIR}...")
    for i in range(1, 11):
        generate_human_resume(i)
        generate_ai_resume(i)
    print("Done! Generated 10 Human and 10 AI resumes.")

if __name__ == "__main__":
    main()
