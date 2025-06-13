import os
from collections import Counter
import re
import spacy
from flask import flash
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text as extract_pdf
from docx import Document
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_pdf(file_path)
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def extract_name(text):
    """
    Basic name extractor: assumes name is the first non-empty line.
    You can replace this with spaCy or any NER model for more accuracy.
    """
    lines = text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line:
            return line
    return "Unknown"

def compute_similarity(text1, text2):
    documents = [text1, text2]
    tfidf = TfidfVectorizer().fit_transform(documents)
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]



@app.route('/')
def home():
    return render_template('index.html')



@app.route('/resume_scanner', methods=['GET', 'POST'])
def resume_scanner():
    if request.method == 'POST':
        jd_text = request.form['jd_text']
        uploaded_files = request.files.getlist("resumes")
        scores = []

        for file in uploaded_files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            resume_text = extract_text(file_path)
            candidate_name = extract_name(resume_text)
            score = compute_similarity(jd_text, resume_text)

            scores.append((candidate_name, round(score * 100, 2)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return render_template('resume_scanner.html', scores=scores)

    return render_template('resume_scanner.html')


nlp = spacy.load("en_core_web_sm")
@app.route('/gap_identifier', methods=['GET', 'POST'])
def gap_identifier():
    gap_results = {}
    if request.method == 'POST':
        jd_text = request.form['jd_text']
        resume_files = request.files.getlist('resume_files')

        def extract_skills(text):
            doc = nlp(text)
            return set(
                token.text.lower()
                for token in doc
                if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2
            )

        jd_skills = extract_skills(jd_text)

        for resume_file in resume_files:
            filename = secure_filename(resume_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_file.save(file_path)

            resume_text = extract_text(file_path)
            resume_skills = extract_skills(resume_text)
            missing_skills = sorted(list(jd_skills - resume_skills))

            gap_results[filename] = missing_skills

    return render_template('gap_identifier.html', gap_results=gap_results)


DOMAIN_TECH = {
    "web development": ["React", "Node.js", "Express", "MongoDB", "Next.js", "Tailwind", "REST APIs", "GraphQL", "Docker"],
    "data science": ["Python", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "Deep Learning", "NLP", "Jupyter"],
    "cloud/devops": ["AWS", "Azure", "CI/CD", "Docker", "Kubernetes", "Terraform", "Jenkins", "Cloud Security"],
    "android": ["Flutter", "Dart", "Kotlin", "Jetpack Compose", "Firebase"],
    "general tech": ["Git", "Agile", "Linux", "OOP", "SQL", "REST"]
}

def detect_domain(resume_text):
    lower_text = resume_text.lower()
    counts = {}

    for domain, techs in DOMAIN_TECH.items():
        count = sum(tech.lower() in lower_text for tech in techs)
        counts[domain] = count

    best_match = max(counts, key=counts.get)
    return best_match if counts[best_match] > 0 else "general tech"

@app.route('/resume_optimizer', methods=['GET', 'POST'])
def resume_optimizer():
    suggestions = []
    if request.method == 'POST':
        resume_file = request.files['resume_file']
        filename = secure_filename(resume_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(file_path)

        resume_text = extract_text(file_path)
        doc = nlp(resume_text)
        domain = detect_domain(resume_text)
        domain_tech = DOMAIN_TECH[domain]

        # Trending tech suggestions
        tech_missing = [tech for tech in domain_tech if tech.lower() not in resume_text.lower()]
        if tech_missing:
            suggestions.append(f"Consider adding relevant modern tech skills like: {', '.join(tech_missing[:5])}.")

        # Grammar & vague terms
        vague = ["worked", "did", "helped", "responsible for", "made"]
        vague_found = [token.text for token in doc if token.lemma_ in vague]
        if vague_found:
            suggestions.append(f"Replace weak verbs like {', '.join(set(vague_found))} with powerful actions (e.g., 'engineered', 'led', 'developed').")

        # Missing structure
        if "project" not in resume_text.lower():
            suggestions.append("Add a 'Projects' section to highlight hands-on experience.")
        if "certification" not in resume_text.lower():
            suggestions.append("List any technical certifications to boost credibility.")

        # Outdated or irrelevant stuff
        outdated = ["Windows XP", "MS Paint", "typing speed"]
        used_outdated = [kw for kw in outdated if kw.lower() in resume_text.lower()]
        if used_outdated:
            suggestions.append(f"Remove outdated mentions: {', '.join(used_outdated)}.")

        # Repetitive words
        word_freq = Counter([token.text.lower() for token in doc if token.is_alpha])
        repeated = [w for w, c in word_freq.items() if c > 5 and len(w) > 3]
        if repeated:
            suggestions.append(f"Avoid overusing: {', '.join(repeated)} — try varied language.")

        # Strong final advice if no suggestions
        if len(suggestions) == 0:
            suggestions.append("Impressive resume! Just fine-tune with more quantified achievements and trend-specific tools.")

    return render_template('resume_optimizer.html', suggestions=suggestions)


@app.route('/resume_score_checker', methods=['GET', 'POST'])
def resume_score_checker():
    score = 0
    suggestions = []
    trending_tech = ["AI", "Machine Learning", "Deep Learning", "Cloud Computing", "AWS", "Azure", "DevOps", "Kubernetes", "Docker"]

    if request.method == 'POST':
        resume_file = request.files['resume_file']
        filename = secure_filename(resume_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(file_path)

        resume_text = extract_text(file_path)
        doc = nlp(resume_text)

        # === SCORE CALCULATION LOGIC ===

        # 1. Grammar and spelling issues
        if len([sent for sent in doc.sents if len(sent.text.split()) < 3]) > 3:
            suggestions.append("Avoid using too many short or incomplete sentences.")

        # 2. Action verbs check (example list)
        action_verbs = ["managed", "developed", "led", "created", "designed", "implemented", "improved"]
        if not any(verb in resume_text.lower() for verb in action_verbs):
            suggestions.append("Use action verbs like 'developed', 'implemented', or 'managed' to describe your experience.")

        # 3. Structure and formatting
        if len(resume_text) < 300:
            suggestions.append("Your resume seems short. Consider adding more details about your projects or experience.")

        # 4. Tech skills & trending technologies
        tech_in_resume = [tech for tech in trending_tech if tech.lower() in resume_text.lower()]
        missing_tech = list(set(trending_tech) - set(tech_in_resume))
        if len(tech_in_resume) < 3:
            suggestions.append(f"Consider adding trending technologies relevant to your domain: {', '.join(missing_tech[:3])}")

        # Final score estimate based on completeness
        score = 50
        if not suggestions:
            score = 95
        else:
            score += (5 * len(tech_in_resume)) - (10 * len(suggestions))
            score = max(30, min(score, 95))  # Bound score between 30 and 95

    return render_template('resume_score_checker.html', score=score, suggestions=suggestions)

@app.route('/compare_resumes', methods=['GET', 'POST'])
def compare_resumes():
    resume_data = []
    compare_count = 2  # default to 2

    if request.method == 'POST':
        compare_count = int(request.form.get('compare_count', 2))
        uploaded_files = request.files.getlist("resume_files")

        if len(uploaded_files) < compare_count:
            return render_template('compare_resumes.html', error="Please upload enough resumes.", compare_count=compare_count)

        trending_tech = ["AI", "Machine Learning", "Deep Learning", "Cloud Computing", "AWS", "Azure", "DevOps", "Docker", "Kubernetes"]
        action_verbs = ["developed", "led", "designed", "built", "implemented", "improved", "created", "managed"]

        for file in uploaded_files[:compare_count]:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            resume_text = extract_text(file_path)
            extracted_name = extract_name(resume_text)
            doc = nlp(resume_text)

            strengths = []
            weaknesses = []

            # 1. Check length and content
            if len(resume_text) > 500:
                strengths.append("Resume has good content coverage.")
            else:
                weaknesses.append("Resume may lack detailed experience or project descriptions.")

            # 2. Action verbs
            if any(verb in resume_text.lower() for verb in action_verbs):
                strengths.append("Uses action-oriented language.")
            else:
                weaknesses.append("Lacks action verbs. Try verbs like 'developed', 'led', etc.")

            # 3. Grammar check
            short_sents = [sent for sent in doc.sents if len(sent.text.split()) < 3]
            if len(short_sents) > 3:
                weaknesses.append("Contains too many short or incomplete sentences.")
            else:
                strengths.append("Maintains sentence structure well.")

            # 4. Trending Tech
            tech_found = [tech for tech in trending_tech if tech.lower() in resume_text.lower()]
            if len(tech_found) >= 3:
                strengths.append(f"Includes trending tech: {', '.join(tech_found)}")
            else:
                weaknesses.append(f"Missing some trending tech like: {', '.join(set(trending_tech) - set(tech_found))}")

            resume_data.append({
                "name": extracted_name,
                "strengths": strengths,
                "weaknesses": weaknesses
            })

    return render_template("compare_resumes.html", resume_data=resume_data, compare_count=compare_count)


def extract_contact_info(text):
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    phone = re.findall(r'\+?\d[\d\-]{8,12}\d', text)
    return email[0] if email else "", phone[0] if phone else ""

def extract_education(text):
    education_keywords = ['b.tech', 'm.tech', 'bachelor', 'master', 'degree', 'phd', 'university', 'college']
    lines = text.lower().splitlines()
    edu_lines = [line.strip() for line in lines if any(k in line for k in education_keywords)]
    return edu_lines

def extract_experience(text):
    experience_keywords = ['experience', 'intern', 'worked', 'project', 'developed', 'engineer']
    lines = text.lower().splitlines()
    exp_lines = [line.strip() for line in lines if any(k in line for k in experience_keywords)]
    return exp_lines

def extract_skills(text):
    skills = []
    predefined_skills = ['python', 'java', 'c++', 'sql', 'html', 'css', 'javascript', 'react', 'node', 'aws', 'azure', 'docker', 'ml', 'ai', 'devops']
    for skill in predefined_skills:
        if skill.lower() in text.lower():
            skills.append(skill)
    return skills

@app.route('/resume_data_extractor', methods=['GET', 'POST'])
def resume_data_extractor():
    data = {}
    if request.method == 'POST':
        file = request.files['resume_file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = extract_text(file_path)
        doc = nlp(text)

        data['name'] = extract_name(text)
        data['email'], data['phone'] = extract_contact_info(text)
        data['skills'] = extract_skills(text)
        data['education'] = extract_education(text)
        data['experience'] = extract_experience(text)

    return render_template('resume_data_extractor.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)