from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import pdfplumber
import docx
from openai import OpenAI
import json
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import timedelta, datetime
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- DATABASE SETUP ----------------
def init_db():
    conn = sqlite3.connect('resume_analyzer.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  full_name TEXT,
                  email TEXT UNIQUE,
                  password TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  theme_preference TEXT DEFAULT 'light')''')
    
    # Create resume_data table
    c.execute('''CREATE TABLE IF NOT EXISTS resume_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  data TEXT,
                  feedback TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create chat_history table
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  message TEXT,
                  response TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create job_comparisons table
    c.execute('''CREATE TABLE IF NOT EXISTS job_comparisons
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  job_description TEXT,
                  analysis TEXT,
                  match_score REAL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Insert demo user if not exists
    c.execute("SELECT * FROM users WHERE username = 'demo'")
    if not c.fetchone():
        hashed_password = generate_password_hash('demo123')
        c.execute("INSERT INTO users (username, full_name, email, password) VALUES (?, ?, ?, ?)",
                  ('demo', 'Demo User', 'demo@example.com', hashed_password))
    
    conn.commit()
    conn.close()

init_db()

# ---------------- CONFIG ----------------
API_KEY = os.getenv("HF_API_KEY", "your-huggingface-api-key-here")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://router.huggingface.co/v1"
)

MODEL_NAME = "google/gemma-4-31B-it:novita"

# ---------------- DECORATORS ----------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('landing'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text(file_path, file_ext):
    if file_ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext == 'docx':
        return extract_text_from_docx(file_path)
    else:  # txt
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

def truncate_text(text, max_chars=15000):
    if len(text) > max_chars:
        return text[:max_chars] + "... [truncated due to length]"
    return text

# ---------------- NEW: RESUME CONTENT VALIDATION ----------------
def is_valid_resume_content(text):
    """
    Validate if the extracted text is actually a resume
    Returns: (is_valid, message, confidence_score)
    """
    if not text or len(text.strip()) < 100:
        return False, "File is too short to be a valid resume (minimum 100 characters required).", 0
    
    text_lower = text.lower()
    
    # Key resume sections to check
    resume_indicators = {
        'experience': ['experience', 'work experience', 'employment', 'work history', 'professional experience', 'work'],
        'education': ['education', 'academic', 'qualifications', 'degree', 'university', 'college', 'school', 'bachelor', 'master'],
        'skills': ['skills', 'technical skills', 'core competencies', 'expertise', 'technologies'],
        'contact': ['@', 'phone', 'email', 'contact', 'mobile', 'telephone']
    }
    
    found_sections = []
    for section, keywords in resume_indicators.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_sections.append(section)
                break
    
    # Count unique sections found
    unique_sections = set(found_sections)
    section_count = len(unique_sections)
    
    # Calculate confidence score based on sections found
    confidence = (section_count / len(resume_indicators)) * 100
    
    # Check for common non-resume patterns
    non_resume_patterns = [
        r'invoice', r'receipt', r'bill to', r'amount due',
        r'table of contents', r'chapter \d+', r'copyright \d+',
        r'isbn', r'journal', r'conference', r'abstract'
    ]
    
    for pattern in non_resume_patterns:
        if re.search(pattern, text_lower):
            return False, f"This appears to be a document containing '{pattern}' which is not typical for a resume.", confidence
    
    # Validation rules
    if section_count >= 2:
        # Has at least 2 key sections
        if 'contact' in unique_sections:
            return True, "Valid resume detected.", confidence
        else:
            return True, "Valid resume detected (contains experience, education, or skills).", confidence
    elif section_count == 1:
        return True, "This may be a resume but is missing some key sections (experience, education, or skills).", confidence
    else:
        return False, "This does not appear to be a resume. Missing key sections like experience, education, or skills.", confidence

# ---------------- RESUME FEEDBACK ----------------
def analyze_resume_quality(resume_data):
    prompt = f"""
You are an expert resume reviewer. Analyze this resume data and provide detailed feedback.

Resume Data:
{json.dumps(resume_data, indent=2)}

Provide a comprehensive analysis in the following JSON format:
{{
    "overall_score": 0,
    "format_score": 0,
    "content_score": 0,
    "ats_score": 0,
    "strengths": ["List of resume strengths"],
    "weaknesses": ["List of areas for improvement"],
    "format_feedback": {{
        "structure": "Feedback on resume structure",
        "length": "Feedback on resume length",
        "readability": "Feedback on readability",
        "consistency": "Feedback on formatting consistency"
    }},
    "content_feedback": {{
        "personal_info": "Feedback on personal info section",
        "skills": "Feedback on skills section",
        "experience": "Feedback on experience section",
        "education": "Feedback on education section",
        "projects": "Feedback on projects section"
    }},
    "recommendations": [
        {{
            "category": "Category",
            "suggestion": "Specific improvement suggestion",
            "priority": "high/medium/low"
        }}
    ],
    "color_code": "green/yellow/red based on overall score"
}}

Be objective and constructive. Format feedback should consider layout, spacing, and visual appeal.
"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            result = result[json_start:json_end]
        return json.loads(result)
    except Exception as e:
        print(f"Error in resume analysis: {e}")
        return {
            "overall_score": 70,
            "format_score": 70,
            "content_score": 70,
            "ats_score": 70,
            "strengths": ["Good basic structure"],
            "weaknesses": ["Could be more detailed"],
            "color_code": "yellow"
        }

# ---------------- ENHANCED SKILL EXTRACTION FUNCTIONS ----------------
def extract_skills_with_phrases(text):
    """Extract skills from text including phrases like 'Python programming'"""
    skills_found = set()
    
    # Common skill phrases patterns
    phrase_patterns = [
        # Programming languages with common suffixes
        r'\b(Python)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(Java)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(JavaScript)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(TypeScript)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(HTML)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer|coding)\b',
        r'\b(CSS)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer|styling)\b',
        r'\b(C\+\+|C#|C)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(Ruby)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(PHP)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(Swift)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(Kotlin)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(Go)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(Rust)\s+(?:programming|development|experience|expertise|knowledge|skills?|developer|engineer)\b',
        
        # Frameworks
        r'\b(React)\s+(?:development|experience|expertise|knowledge|skills?|developer|engineer|framework)\b',
        r'\b(Angular)\s+(?:development|experience|expertise|knowledge|skills?|developer|engineer|framework)\b',
        r'\b(Vue)\s+(?:development|experience|expertise|knowledge|skills?|developer|engineer|framework)\b',
        r'\b(Django)\s+(?:development|experience|expertise|knowledge|skills?|developer|engineer|framework)\b',
        r'\b(Flask)\s+(?:development|experience|expertise|knowledge|skills?|developer|engineer|framework)\b',
        r'\b(Spring)\s+(?:development|experience|expertise|knowledge|skills?|developer|engineer|framework)\b',
        r'\b(Laravel)\s+(?:development|experience|expertise|knowledge|skills?|developer|engineer|framework)\b',
        r'\b(Bootstrap)\s+(?:development|experience|expertise|knowledge|skills?|developer|engineer|framework)\b',
        r'\b(Tailwind)\s+(?:development|experience|expertise|knowledge|skills?|developer|engineer|framework)\b',
        
        # Cloud & DevOps
        r'\b(AWS)\s+(?:cloud|services|development|experience|expertise|knowledge|skills?|engineer|architect)\b',
        r'\b(Azure)\s+(?:cloud|services|development|experience|expertise|knowledge|skills?|engineer|architect)\b',
        r'\b(GCP)\s+(?:cloud|services|development|experience|expertise|knowledge|skills?|engineer|architect)\b',
        r'\b(Docker)\s+(?:containerization|experience|expertise|knowledge|skills?|engineer|devops)\b',
        r'\b(Kubernetes)\s+(?:orchestration|experience|expertise|knowledge|skills?|engineer|devops)\b',
        r'\b(Jenkins)\s+(?:experience|expertise|knowledge|skills?|engineer|ci/cd)\b',
        r'\b(Git)\s+(?:version control|experience|expertise|knowledge|skills?)\b',
        
        # Databases
        r'\b(SQL)\s+(?:database|querying|experience|expertise|knowledge|skills?|developer|engineer)\b',
        r'\b(MySQL)\s+(?:database|experience|expertise|knowledge|skills?|developer)\b',
        r'\b(PostgreSQL)\s+(?:database|experience|expertise|knowledge|skills?|developer)\b',
        r'\b(MongoDB)\s+(?:database|experience|expertise|knowledge|skills?|developer)\b',
        r'\b(Redis)\s+(?:database|caching|experience|expertise|knowledge|skills?)\b',
        
        # Data Science & ML
        r'\b(Machine Learning)\s+(?:experience|expertise|knowledge|skills?|engineer|scientist)\b',
        r'\b(Deep Learning)\s+(?:experience|expertise|knowledge|skills?|engineer|scientist)\b',
        r'\b(AI|Artificial Intelligence)\s+(?:experience|expertise|knowledge|skills?|engineer|scientist)\b',
        r'\b(Data Science)\s+(?:experience|expertise|knowledge|skills?|engineer|scientist)\b',
        r'\b(TensorFlow)\s+(?:experience|expertise|knowledge|skills?)\b',
        r'\b(PyTorch)\s+(?:experience|expertise|knowledge|skills?)\b',
        r'\b(Pandas)\s+(?:experience|expertise|knowledge|skills?)\b',
        r'\b(NumPy)\s+(?:experience|expertise|knowledge|skills?)\b'
    ]
    
    # Extract skills from phrases
    for pattern in phrase_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                skills_found.add(match[0])
            else:
                skills_found.add(match)
    
    # Also extract standalone skills
    standalone_skills = extract_skills_from_text(text)
    skills_found.update(standalone_skills)
    
    return list(skills_found)

def extract_all_skills_from_resume(resume_data):
    """
    Extract all skills from the nested resume data structure
    This function properly handles HTML and CSS as programming languages
    """
    all_skills = set()
    
    try:
        # If resume_data is a dictionary
        if isinstance(resume_data, dict):
            
            # 1. Extract from skills section (nested structure)
            if 'skills' in resume_data and isinstance(resume_data['skills'], dict):
                skills_section = resume_data['skills']
                
                # Programming languages - explicitly check for HTML/CSS
                if 'programming_languages' in skills_section and isinstance(skills_section['programming_languages'], list):
                    for skill in skills_section['programming_languages']:
                        if skill and isinstance(skill, str):
                            all_skills.add(skill.strip())
                            all_skills.add(skill.lower().strip())
                
                # Technical skills - often HTML/CSS appear here
                if 'technical_skills' in skills_section and isinstance(skills_section['technical_skills'], list):
                    for skill in skills_section['technical_skills']:
                        if skill and isinstance(skill, str):
                            all_skills.add(skill.strip())
                            all_skills.add(skill.lower().strip())
                
                # Frameworks
                if 'frameworks' in skills_section and isinstance(skills_section['frameworks'], list):
                    for skill in skills_section['frameworks']:
                        if skill and isinstance(skill, str):
                            all_skills.add(skill.strip())
                            all_skills.add(skill.lower().strip())
                
                # Databases
                if 'databases' in skills_section and isinstance(skills_section['databases'], list):
                    for skill in skills_section['databases']:
                        if skill and isinstance(skill, str):
                            all_skills.add(skill.strip())
                            all_skills.add(skill.lower().strip())
                
                # Cloud platforms
                if 'cloud_platforms' in skills_section and isinstance(skills_section['cloud_platforms'], list):
                    for skill in skills_section['cloud_platforms']:
                        if skill and isinstance(skill, str):
                            all_skills.add(skill.strip())
                            all_skills.add(skill.lower().strip())
                
                # Tools
                if 'tools' in skills_section and isinstance(skills_section['tools'], list):
                    for skill in skills_section['tools']:
                        if skill and isinstance(skill, str):
                            all_skills.add(skill.strip())
                            all_skills.add(skill.lower().strip())
                
                # Soft skills
                if 'soft_skills' in skills_section and isinstance(skills_section['soft_skills'], list):
                    for skill in skills_section['soft_skills']:
                        if skill and isinstance(skill, str):
                            all_skills.add(skill.strip())
                            all_skills.add(skill.lower().strip())
            
            # 2. Extract from work_experience (technologies_used)
            if 'work_experience' in resume_data and isinstance(resume_data['work_experience'], list):
                for exp in resume_data['work_experience']:
                    if isinstance(exp, dict):
                        # Check for technologies_used field
                        if 'technologies_used' in exp and isinstance(exp['technologies_used'], list):
                            for tech in exp['technologies_used']:
                                if tech and isinstance(tech, str):
                                    all_skills.add(tech.strip())
                                    all_skills.add(tech.lower().strip())
                        
                        # Also check responsibilities text for skills
                        if 'responsibilities' in exp and isinstance(exp['responsibilities'], list):
                            for resp in exp['responsibilities']:
                                if resp and isinstance(resp, str):
                                    # Extract potential skills from text
                                    words = resp.split()
                                    for word in words:
                                        if len(word) > 2 and word[0].isupper():
                                            all_skills.add(word.strip())
            
            # 3. Extract from projects (technologies)
            if 'projects' in resume_data and isinstance(resume_data['projects'], list):
                for proj in resume_data['projects']:
                    if isinstance(proj, dict):
                        if 'technologies' in proj and isinstance(proj['technologies'], list):
                            for tech in proj['technologies']:
                                if tech and isinstance(tech, str):
                                    all_skills.add(tech.strip())
                                    all_skills.add(tech.lower().strip())
            
            # 4. Extract from certifications
            if 'certifications' in resume_data and isinstance(resume_data['certifications'], list):
                for cert in resume_data['certifications']:
                    if isinstance(cert, dict):
                        if 'name' in cert and cert['name']:
                            all_skills.add(cert['name'].strip())
            
            # 5. Extract from education (relevant_coursework)
            if 'education' in resume_data and isinstance(resume_data['education'], list):
                for edu in resume_data['education']:
                    if isinstance(edu, dict):
                        if 'relevant_coursework' in edu and isinstance(edu['relevant_coursework'], list):
                            for course in edu['relevant_coursework']:
                                if course and isinstance(course, str):
                                    all_skills.add(course.strip())
            
            # 6. Extract from additional_sections
            if 'additional_sections' in resume_data and isinstance(resume_data['additional_sections'], dict):
                sections = resume_data['additional_sections']
                if 'section_contents' in sections and isinstance(sections['section_contents'], list):
                    for content in sections['section_contents']:
                        if content and isinstance(content, str):
                            # Simple skill extraction from text
                            words = content.split()
                            for word in words:
                                if len(word) > 2 and word[0].isupper() and word.lower() not in ['the', 'and', 'for', 'with']:
                                    all_skills.add(word.strip())
    
    except Exception as e:
        print(f"Error extracting skills: {e}")
    
    # Convert to list and remove duplicates
    return list(all_skills)

def extract_skills_from_text(text):
    """Enhanced function with HTML/CSS detection"""
    # Common skill patterns - INCLUDING HTML/CSS
    skill_patterns = [
        # Programming Languages
        r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Go|Rust|Scala|HTML|CSS|SASS|SCSS|Less|Stylus)\b',
        
        # Frontend Frameworks & Libraries
        r'\b(?:React|Angular|Vue|Next\.js|Nuxt\.js|Gatsby|Svelte|jQuery|Bootstrap|Tailwind|MaterialUI|SemanticUI|Foundation)\b',
        
        # Backend Frameworks
        r'\b(?:Node\.js|Django|Flask|Spring|Laravel|Rails|Express|ASP\.NET|FastAPI|Ruby on Rails|Phoenix)\b',
        
        # Databases
        r'\b(?:SQL|MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB|Oracle|SQLite|Firebase|Supabase)\b',
        
        # Cloud & DevOps
        r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|CI/CD|Terraform|Ansible|Puppet|Chef|CircleCI|GitHub Actions)\b',
        
        # Data Science & ML
        r'\b(?:Machine Learning|Deep Learning|AI|NLP|Computer Vision|Data Science|Analytics|TensorFlow|PyTorch|Pandas|NumPy|SciPy)\b',
        
        # Project Management & Tools
        r'\b(?:Agile|Scrum|Kanban|JIRA|Confluence|Trello|Asana|ClickUp)\b',
        
        # Soft Skills
        r'\b(?:Communication|Leadership|Problem Solving|Critical Thinking|Teamwork|Time Management|Adaptability|Creativity)\b'
    ]
    
    skills = set()
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Preserve original capitalization for display
            if match.lower() == 'html':
                skills.add('HTML')
            elif match.lower() == 'css':
                skills.add('CSS')
            elif match.lower() in ['sass', 'scss']:
                skills.add(match.upper())
            else:
                skills.add(match)
    
    return list(skills)

def normalize_skill(skill):
    """Normalize skill names for better matching - UPDATED with HTML/CSS"""
    skill_lower = skill.lower().strip()
    # Common variations
    variations = {
        # Programming Languages
        'python': 'Python',
        'java': 'Java',
        'javascript': 'JavaScript',
        'js': 'JavaScript',
        'typescript': 'TypeScript',
        'ts': 'TypeScript',
        'c++': 'C++',
        'cpp': 'C++',
        'c#': 'C#',
        'csharp': 'C#',
        'html': 'HTML',
        'html5': 'HTML',
        'css': 'CSS',
        'css3': 'CSS',
        'sass': 'SASS',
        'scss': 'SCSS',
        
        # Frameworks
        'react': 'React',
        'react.js': 'React',
        'angular': 'Angular',
        'angular.js': 'Angular',
        'vue': 'Vue.js',
        'vue.js': 'Vue.js',
        'node': 'Node.js',
        'node.js': 'Node.js',
        'django': 'Django',
        'flask': 'Flask',
        'bootstrap': 'Bootstrap',
        'tailwind': 'Tailwind CSS',
        'tailwind css': 'Tailwind CSS',
        'materialui': 'Material-UI',
        'mui': 'Material-UI',
        
        # Databases
        'sql': 'SQL',
        'mysql': 'MySQL',
        'postgresql': 'PostgreSQL',
        'postgres': 'PostgreSQL',
        'mongodb': 'MongoDB',
        'mongo': 'MongoDB',
        
        # Cloud
        'aws': 'AWS',
        'amazon web services': 'AWS',
        'azure': 'Azure',
        'gcp': 'GCP',
        'google cloud': 'GCP',
        'docker': 'Docker',
        'kubernetes': 'Kubernetes',
        'k8s': 'Kubernetes',
        'git': 'Git',
        'github': 'GitHub'
    }
    
    if skill_lower in variations:
        return variations[skill_lower]
    
    # Special handling for HTML/CSS variations
    if 'html' in skill_lower:
        return 'HTML'
    if 'css' in skill_lower and 'sass' not in skill_lower and 'scss' not in skill_lower:
        return 'CSS'
    
    return skill.title()

def categorize_skill(skill):
    """Categorize skill into appropriate group"""
    skill_lower = skill.lower()
    
    # Programming Languages (including HTML/CSS)
    programming_languages = ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 
                            'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'html', 'css', 
                            'sass', 'scss', 'less']
    
    # Frameworks
    frameworks = ['react', 'angular', 'vue', 'next.js', 'django', 'flask', 'spring', 
                 'laravel', 'rails', 'express', 'bootstrap', 'tailwind', 'material-ui',
                 'jquery', 'svelte', 'gatsby', 'nuxt.js']
    
    # Databases
    databases = ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 
                'cassandra', 'dynamodb', 'oracle', 'sqlite', 'firebase', 'supabase']
    
    # Cloud & DevOps
    cloud_devops = ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 
                   'ci/cd', 'terraform', 'ansible', 'puppet', 'chef', 'github actions']
    
    # Data Science
    data_science = ['machine learning', 'deep learning', 'ai', 'nlp', 'computer vision', 
                   'data science', 'analytics', 'tensorflow', 'pytorch', 'pandas', 'numpy']
    
    # Tools
    tools = ['jira', 'confluence', 'trello', 'asana', 'clickup', 'slack', 'teams',
             'vscode', 'visual studio', 'eclipse', 'intellij', 'pycharm']
    
    # Soft Skills
    soft_skills = ['communication', 'leadership', 'problem solving', 'critical thinking', 
                  'teamwork', 'time management', 'adaptability', 'creativity', 'collaboration']
    
    for pl in programming_languages:
        if pl in skill_lower:
            return 'programming_languages'
    
    for fw in frameworks:
        if fw in skill_lower:
            return 'frameworks'
    
    for db in databases:
        if db in skill_lower:
            return 'databases'
    
    for cd in cloud_devops:
        if cd in skill_lower:
            return 'cloud_platforms'
    
    for ds in data_science:
        if ds in skill_lower:
            return 'data_science'
    
    for tool in tools:
        if tool in skill_lower:
            return 'tools'
    
    for soft in soft_skills:
        if soft in skill_lower:
            return 'soft_skills'
    
    return 'technical_skills'  # Default category

# ---------------- COSINE SIMILARITY MATCHING ----------------
def resume_to_text(resume_data):
    """Convert resume data structure to clean text for cosine similarity"""
    text_parts = []

    # Personal Info (summary and objective)
    personal = resume_data.get('personal_info', {})
    if personal.get('summary'):
        text_parts.append(personal.get('summary'))
    if personal.get('objective'):
        text_parts.append(personal.get('objective'))

    # Skills
    skills = resume_data.get('skills', {})
    for key, value in skills.items():
        if isinstance(value, list):
            text_parts.extend(value)

    # Experience (with achievements added)
    for exp in resume_data.get('work_experience', []):
        text_parts.extend(exp.get('responsibilities', []))
        text_parts.extend(exp.get('technologies_used', []))
        text_parts.extend(exp.get('achievements', []))

    # Projects
    for proj in resume_data.get('projects', []):
        text_parts.append(proj.get('description', ''))
        text_parts.extend(proj.get('technologies', []))

    # Certifications
    for cert in resume_data.get('certifications', []):
        if isinstance(cert, dict):
            text_parts.append(cert.get('name', ''))

    # Education coursework
    for edu in resume_data.get('education', []):
        if isinstance(edu, dict):
            text_parts.extend(edu.get('relevant_coursework', []))

    return " ".join(text_parts)

def clean_text(text):
    """Clean text for better similarity comparison"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text

def normalize_keywords(text):
    """Normalize keywords to fix semantic mismatch"""
    replacements = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "js": "javascript",
        "nodejs": "node js",
        "reactjs": "react",
        "vuejs": "vue",
        "nextjs": "next",
        "ts": "typescript",
        "html5": "html",
        "css3": "css"
    }
    
    text_lower = text.lower()
    for k, v in replacements.items():
        text_lower = text_lower.replace(k, v)
    
    return text_lower

def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two texts with improved TF-IDF"""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),   # big improvement for matching phrases
        max_features=5000
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity * 100, 2)
    except Exception as e:
        print("Cosine error:", e)
        return 0

def extract_job_requirements(job_description):
    """Extract requirements from job description using AI"""
    prompt = f"""
Extract key requirements from this job description. Return in JSON format.

Job Description:
{job_description}

IMPORTANT INSTRUCTIONS:
1. Extract skills by identifying both single skills (like "Python") and skill phrases (like "Python programming", "Python development")
2. For each skill, extract the base skill name only (e.g., from "Python programming" extract "Python")
3. Include both required and preferred skills
4. Pay attention to skill variations and synonyms
5. Include soft skills like communication, leadership, etc.

Return format:
{{
    "required_skills": [],
    "preferred_skills": [],
    "required_experience": "",
    "education_requirements": "",
    "key_responsibilities": [],
    "soft_skills": []
}}

For skills, extract clean skill names without extra words. Examples:
- "Python programming" -> "Python"
- "JavaScript development" -> "JavaScript"  
- "HTML/CSS" -> "HTML" and "CSS" separately
- "React.js experience" -> "React"
- "AWS cloud services" -> "AWS"
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            result = result[json_start:json_end]
        return json.loads(result)
    except Exception as e:
        print(f"Error extracting job requirements: {e}")
        return {
            "required_skills": [],
            "preferred_skills": [],
            "required_experience": "",
            "education_requirements": "",
            "key_responsibilities": [],
            "soft_skills": []
        }

# ---------------- FIXED ATS SCORE CALCULATION ----------------
def calculate_ats_score(matching_skills, required_skills, preferred_skills):
    """
    Calculate ATS Score based on formula:
    ATS Score = (Matched Keywords / Total JD Keywords) * 100
    
    FIXED: Now properly validates that matched skills are actually in job requirements
    """
    # Convert all to lowercase for proper matching
    required_lower = [s.lower() for s in required_skills]
    preferred_lower = [s.lower() for s in preferred_skills]
    all_job_keywords = set(required_lower + preferred_lower)
    
    # Only count skills that actually exist in job requirements
    valid_matches = []
    for skill in matching_skills:
        if skill.lower() in all_job_keywords:
            valid_matches.append(skill)
    
    total_keywords = len(all_job_keywords)
    matched_keywords = len(valid_matches)
    
    if total_keywords == 0:
        return 0
    
    ats_score = round((matched_keywords / total_keywords) * 100, 2)
    return ats_score

# ---------------- NEW: VALIDATE AND CLEAN MATCHES ----------------
def validate_and_clean_matches(matching_skills, required_skills, preferred_skills):
    """
    Clean up matching skills to ensure they only include valid job keywords
    This fixes the ATS score calculation bug
    """
    required_lower = [s.lower() for s in required_skills]
    preferred_lower = [s.lower() for s in preferred_skills]
    all_job_lower = set(required_lower + preferred_lower)
    
    # Valid matches only - skills that exist in job requirements
    valid_matches = []
    for skill in matching_skills:
        if skill.lower() in all_job_lower:
            valid_matches.append(skill)
        else:
            print(f"Warning: '{skill}' matched but not in job requirements - removing from matches")
    
    # Find missing required skills
    missing_required = []
    for req in required_skills:
        if req.lower() not in [s.lower() for s in valid_matches]:
            missing_required.append(req)
    
    # Find missing preferred skills
    missing_preferred = []
    for pref in preferred_skills:
        if pref.lower() not in [s.lower() for s in valid_matches]:
            missing_preferred.append(pref)
    
    return valid_matches, missing_required, missing_preferred

# ---------------- LLM-BASED SKILL MATCHING ----------------
def match_skills_with_llm(resume_skills, job_required_skills, job_preferred_skills):
    """
    Use LLM to intelligently match skills between resume and job description
    This handles synonyms, variations, and contextual matching
    """
    if not job_required_skills and not job_preferred_skills:
        return [], [], []
    
    # Limit skills to avoid token overflow
    resume_skills_sample = resume_skills[:50] if len(resume_skills) > 50 else resume_skills
    
    prompt = f"""
You are an expert HR professional matching candidate skills with job requirements.

Resume Skills (from candidate's resume):
{json.dumps(resume_skills_sample, indent=2)}

Job Required Skills (must have):
{json.dumps(job_required_skills, indent=2)}

Job Preferred Skills (nice to have):
{json.dumps(job_preferred_skills, indent=2)}

Your task is to intelligently match the resume skills with the job skills.

Consider:
1. Synonyms (e.g., "JavaScript" matches "JS", "ECMAScript")
2. Related technologies (e.g., "React" is related to "Frontend Development")
3. Variations (e.g., "Python programming" matches "Python")
4. Partial matches (e.g., "Machine Learning" matches "ML")
5. Context understanding (e.g., "Cloud" matches "AWS", "Azure", "GCP")
6. Similar skill groups (e.g., "Data Analysis" matches "Data Science")

IMPORTANT: Only return skills that EXACTLY match or are CLEAR SYNONYMS of skills in the job requirements lists.
Use the EXACT skill names as they appear in the job requirements.

Return ONLY a JSON object in this exact format:
{{
    "matching_skills": [
        "skill1",
        "skill2"
    ],
    "missing_required": [
        "required_skill1",
        "required_skill2"
    ],
    "missing_preferred": [
        "preferred_skill1",
        "preferred_skill2"
    ],
    "explanations": {{
        "skill1": "Why this matches",
        "required_skill1": "Why this is missing"
    }}
}}

Rules:
- Only include skills in matching_skills if the resume has a clear match
- Be strict about required skills - if the resume doesn't have it, add to missing_required
- Be flexible with preferred skills - only add to missing_preferred if clearly missing
- Provide brief explanations for key matches and misses
- Use the skill names exactly as they appear in the job skills lists

Make sure to output valid JSON only, no other text.
"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500
        )
        
        result = response.choices[0].message.content
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            result = result[json_start:json_end]
        
        llm_match_result = json.loads(result)
        
        return (
            llm_match_result.get('matching_skills', []),
            llm_match_result.get('missing_required', []),
            llm_match_result.get('missing_preferred', [])
        )
    except Exception as e:
        print(f"Error in LLM skill matching: {e}")
        # Fallback to traditional matching
        return fallback_skill_matching(resume_skills, job_required_skills, job_preferred_skills)

def fallback_skill_matching(resume_skills, job_required_skills, job_preferred_skills):
    """Fallback matching function if LLM fails"""
    matching_skills = []
    missing_required = []
    missing_preferred = []
    
    resume_skills_lower = [s.lower() for s in resume_skills]
    
    # Match required skills
    for skill in job_required_skills:
        skill_lower = skill.lower()
        found = False
        for res_skill in resume_skills_lower:
            if skill_lower in res_skill or res_skill in skill_lower:
                found = True
                break
        if found:
            matching_skills.append(skill)
        else:
            missing_required.append(skill)
    
    # Match preferred skills
    for skill in job_preferred_skills:
        skill_lower = skill.lower()
        found = False
        for res_skill in resume_skills_lower:
            if skill_lower in res_skill or res_skill in skill_lower:
                found = True
                break
        if found:
            if skill not in matching_skills:
                matching_skills.append(skill)
        else:
            missing_preferred.append(skill)
    
    return matching_skills, missing_required, missing_preferred

# ---------------- COURSE RECOMMENDATIONS ----------------
def get_course_recommendations(skill_gaps):
    """Get course recommendations based on skill gaps - UPDATED with HTML/CSS courses"""
    
    course_database = {
        "HTML": [
            {"name": "HTML5 Complete Course", "platform": "Udemy", "url": "https://www.udemy.com/course/html5-fundamentals-for-beginners/", "duration": "6 hours"},
            {"name": "Introduction to HTML5", "platform": "Coursera", "url": "https://www.coursera.org/learn/html", "duration": "3 weeks"},
            {"name": "HTML & CSS Tutorials", "platform": "freeCodeCamp", "url": "https://www.freecodecamp.org/", "duration": "40 hours"}
        ],
        "CSS": [
            {"name": "CSS - The Complete Guide", "platform": "Udemy", "url": "https://www.udemy.com/course/css-the-complete-guide-incl-flexbox-grid-sass/", "duration": "22 hours"},
            {"name": "Advanced CSS and Sass", "platform": "Udemy", "url": "https://www.udemy.com/course/advanced-css-and-sass/", "duration": "28 hours"},
            {"name": "Responsive Web Design", "platform": "freeCodeCamp", "url": "https://www.freecodecamp.org/", "duration": "300 hours"}
        ],
        "SASS": [
            {"name": "Sass Complete Course", "platform": "Udemy", "url": "https://www.udemy.com/course/sass-the-complete-guide-to-learn-sass/", "duration": "4 hours"}
        ],
        "Bootstrap": [
            {"name": "Bootstrap 5 Complete Course", "platform": "Udemy", "url": "https://www.udemy.com/course/bootstrap-4-tutorials/", "duration": "10 hours"}
        ],
        "Tailwind CSS": [
            {"name": "Tailwind CSS Complete Course", "platform": "Udemy", "url": "https://www.udemy.com/course/tailwind-css-from-scratch/", "duration": "12 hours"}
        ],
        "Python": [
            {"name": "Complete Python Bootcamp", "platform": "Udemy", "url": "https://www.udemy.com/course/complete-python-bootcamp/", "duration": "22 hours"},
            {"name": "Python for Everybody", "platform": "Coursera", "url": "https://www.coursera.org/specializations/python", "duration": "8 months"}
        ],
        "JavaScript": [
            {"name": "The Complete JavaScript Course", "platform": "Udemy", "url": "https://www.udemy.com/course/the-complete-javascript-course/", "duration": "28 hours"},
            {"name": "JavaScript Algorithms and Data Structures", "platform": "freeCodeCamp", "url": "https://www.freecodecamp.org/", "duration": "300 hours"}
        ],
        "React": [
            {"name": "React - The Complete Guide", "platform": "Udemy", "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/", "duration": "40 hours"},
            {"name": "Frontend Web Development with React", "platform": "Coursera", "url": "https://www.coursera.org/learn/frontend-react", "duration": "4 months"}
        ],
        "AWS": [
            {"name": "AWS Certified Solutions Architect", "platform": "A Cloud Guru", "url": "https://acloudguru.com/", "duration": "40 hours"},
            {"name": "AWS Fundamentals", "platform": "Coursera", "url": "https://www.coursera.org/learn/aws-fundamentals", "duration": "3 weeks"}
        ],
        "Machine Learning": [
            {"name": "Machine Learning by Andrew Ng", "platform": "Coursera", "url": "https://www.coursera.org/learn/machine-learning", "duration": "11 weeks"},
            {"name": "Deep Learning Specialization", "platform": "Coursera", "url": "https://www.coursera.org/specializations/deep-learning", "duration": "5 months"}
        ],
        "SQL": [
            {"name": "SQL for Data Science", "platform": "Coursera", "url": "https://www.coursera.org/learn/sql-for-data-science", "duration": "4 weeks"},
            {"name": "The Complete SQL Bootcamp", "platform": "Udemy", "url": "https://www.udemy.com/course/the-complete-sql-bootcamp/", "duration": "9 hours"}
        ],
        "Docker": [
            {"name": "Docker Mastery", "platform": "Udemy", "url": "https://www.udemy.com/course/docker-mastery/", "duration": "19 hours"},
            {"name": "Containers & Kubernetes", "platform": "Coursera", "url": "https://www.coursera.org/learn/containers-kubernetes", "duration": "4 weeks"}
        ],
        "Git": [
            {"name": "Git Complete: The definitive guide", "platform": "Udemy", "url": "https://www.udemy.com/course/git-complete/", "duration": "6 hours"},
            {"name": "Version Control with Git", "platform": "Coursera", "url": "https://www.coursera.org/learn/version-control-with-git", "duration": "4 weeks"}
        ]
    }
    
    recommendations = []
    for skill in skill_gaps:
        # Try exact match first
        if skill in course_database:
            recommendations.append({
                "skill": skill,
                "courses": course_database[skill]
            })
        else:
            # Try case-insensitive match
            found = False
            for db_skill in course_database:
                if db_skill.lower() == skill.lower():
                    recommendations.append({
                        "skill": skill,
                        "courses": course_database[db_skill]
                    })
                    found = True
                    break
            
            # If not found, try partial match for common technologies
            if not found:
                skill_lower = skill.lower()
                if 'html' in skill_lower:
                    recommendations.append({
                        "skill": skill,
                        "courses": course_database["HTML"]
                    })
                elif 'css' in skill_lower and 'sass' not in skill_lower and 'scss' not in skill_lower:
                    recommendations.append({
                        "skill": skill,
                        "courses": course_database["CSS"]
                    })
                elif 'python' in skill_lower:
                    recommendations.append({
                        "skill": skill,
                        "courses": course_database["Python"]
                    })
                elif 'javascript' in skill_lower or 'js' in skill_lower:
                    recommendations.append({
                        "skill": skill,
                        "courses": course_database["JavaScript"]
                    })
                elif 'react' in skill_lower:
                    recommendations.append({
                        "skill": skill,
                        "courses": course_database["React"]
                    })
                elif 'sql' in skill_lower:
                    recommendations.append({
                        "skill": skill,
                        "courses": course_database["SQL"]
                    })
                elif 'docker' in skill_lower:
                    recommendations.append({
                        "skill": skill,
                        "courses": course_database["Docker"]
                    })
                elif 'git' in skill_lower:
                    recommendations.append({
                        "skill": skill,
                        "courses": course_database["Git"]
                    })
    
    return recommendations

# ---------------- SECTION EXTRACTION ----------------
def extract_resume_sections(resume_text):
    truncated_text = truncate_text(resume_text, max_chars=12000)
    
    prompt = f"""
You are a resume parsing expert. Extract information from this resume into a JSON structure.
Pay special attention to HTML, CSS, and frontend technologies - categorize them appropriately.

Return ONLY a valid JSON object with these sections (use empty arrays/objects if section not found):

{{
    "personal_info": {{
        "full_name": "",
        "professional_title": "",
        "email": "",
        "phone": "",
        "location": "",
        "linkedin": "",
        "github": "",
        "portfolio": "",
        "summary": "",
        "objective": ""
    }},
    
    "skills": {{
        "technical_skills": [],
        "soft_skills": [],
        "programming_languages": ["HTML", "CSS", "JavaScript", etc.],
        "frameworks": ["React", "Bootstrap", "Tailwind", etc.],
        "databases": [],
        "cloud_platforms": [],
        "tools": []
    }},
    
    "certifications": [
        {{
            "name": "",
            "issuing_organization": "",
            "issue_date": "",
            "credential_id": ""
        }}
    ],
    
    "education": [
        {{
            "degree": "",
            "field_of_study": "",
            "institution": "",
            "graduation_date": "",
            "gpa": "",
            "honors": "",
            "relevant_coursework": []
        }}
    ],
    
    "work_experience": [
        {{
            "job_title": "",
            "company": "",
            "location": "",
            "start_date": "",
            "end_date": "",
            "current_job": false,
            "responsibilities": [],
            "achievements": [],
            "technologies_used": []
        }}
    ],
    
    "internships": [
        {{
            "title": "",
            "company": "",
            "duration": "",
            "responsibilities": []
        }}
    ],
    
    "projects": [
        {{
            "name": "",
            "description": "",
            "technologies": [],
            "duration": "",
            "link": ""
        }}
    ],
    
    "languages": [
        {{
            "language": "",
            "proficiency": ""
        }}
    ],
    
    "interests": [],
    
    "references": [
        {{
            "name": "",
            "position": "",
            "company": "",
            "email": "",
            "phone": ""
        }}
    ],
    
    "additional_sections": {{
        "section_names": [],
        "section_contents": []
    }}
}}

Resume text:
{truncated_text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )

        result = response.choices[0].message.content
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            result = result[json_start:json_end]
        return json.loads(result)
    except Exception as e:
        print(f"Error: {e}")
        return {}

# ---------------- ENHANCED JOB MATCHING WITH LLM ----------------
def compare_with_job(resume_data, job_description):
    # Extract skills from resume using multiple methods
    resume_skills = extract_all_skills_from_resume(resume_data)
    
    # Also get skills from text as backup
    resume_text_skills = extract_skills_from_text(json.dumps(resume_data))
    resume_skills.extend(resume_text_skills)
    
    # Extract skills from phrases in resume
    resume_phrase_skills = extract_skills_with_phrases(json.dumps(resume_data))
    resume_skills.extend(resume_phrase_skills)
    
    # Remove duplicates
    resume_skills = list(set(resume_skills))
    
    # Extract job requirements with improved skill detection
    job_requirements = extract_job_requirements(job_description)
    
    # Also extract skills from job description using phrase detection
    job_phrase_skills = extract_skills_with_phrases(job_description)
    for skill in job_phrase_skills:
        normalized = normalize_skill(skill)
        # Add to required skills if not already there
        if normalized not in [normalize_skill(s) for s in job_requirements.get('required_skills', [])]:
            if 'required_skills' not in job_requirements:
                job_requirements['required_skills'] = []
            job_requirements['required_skills'].append(normalized)
    
    # Convert resume data to clean text for cosine similarity
    resume_text_clean = resume_to_text(resume_data)
    
    # Normalize keywords and clean text
    resume_text_clean = normalize_keywords(clean_text(resume_text_clean))
    job_description_clean = normalize_keywords(clean_text(job_description))
    
    # Boost skills weight
    skills_text = " ".join(resume_data.get('skills', {}).get('technical_skills', []))
    resume_text_clean = resume_text_clean + " " + (skills_text * 3)
    
    # Calculate cosine similarity with cleaned texts
    similarity_score = calculate_cosine_similarity(resume_text_clean, job_description_clean)
    
    # Use LLM for intelligent skill matching
    matching_skills, missing_required, missing_preferred = match_skills_with_llm(
        resume_skills,
        job_requirements.get('required_skills', []),
        job_requirements.get('preferred_skills', [])
    )
    
    # FIXED: Clean and validate the matches (This fixes the ATS score bug)
    matching_skills, missing_required, missing_preferred = validate_and_clean_matches(
        matching_skills,
        job_requirements.get('required_skills', []),
        job_requirements.get('preferred_skills', [])
    )
    
    # Calculate ATS Score using the cleaned matching skills
    ats_score = calculate_ats_score(matching_skills, 
                                   job_requirements.get('required_skills', []), 
                                   job_requirements.get('preferred_skills', []))
    
    # Calculate final combined score
    total_job_skills = len(job_requirements.get('required_skills', [])) + len(job_requirements.get('preferred_skills', []))
    llm_score = (len(matching_skills) / max(total_job_skills, 1)) * 100
    final_score = (ats_score * 0.5) + (similarity_score * 0.2) + (llm_score * 0.3)
    
    # Get course recommendations for missing skills
    all_missing = missing_required + missing_preferred
    course_recommendations = get_course_recommendations(all_missing[:5])  # Top 5 missing skills
    
    # Get AI analysis for detailed feedback
    prompt = f"""
Based on the resume and job description, provide a detailed analysis.

Resume Data:
{json.dumps(resume_data, indent=2)[:1000]}... (truncated)

Job Requirements:
{json.dumps(job_requirements, indent=2)}

Matching Skills: {matching_skills}
Missing Required Skills: {missing_required}
Missing Preferred Skills: {missing_preferred}

ATS Score: {ats_score}% (based on keyword match)
Cosine Similarity: {similarity_score}% (based on content similarity)

Provide analysis in JSON:
{{
    "summary": "Brief overview of fit",
    "strengths": ["Key strengths for this role"],
    "weaknesses": ["Areas to improve"],
    "experience_analysis": "How experience matches",
    "education_analysis": "How education matches",
    "recommendations": ["Actionable recommendations"],
    "interview_questions": ["Potential interview questions"]
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        
        result = response.choices[0].message.content
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            result = result[json_start:json_end]
        ai_analysis = json.loads(result)
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        ai_analysis = {
            "summary": "Analysis complete",
            "strengths": matching_skills[:3] if matching_skills else ["Your resume has been analyzed"],
            "weaknesses": missing_required[:3] if missing_required else ["Consider adding more skills"],
            "experience_analysis": "Based on your resume content",
            "education_analysis": "Based on your education section",
            "recommendations": ["Consider learning missing skills", "Update your resume with more keywords"],
            "interview_questions": ["Tell me about your experience with " + (matching_skills[0] if matching_skills else "your skills")]
        }
    
    return {
        "match_score": round(final_score, 2),
        "cosine_similarity": similarity_score,
        "ats_score": ats_score,
        "llm_match_score": round(llm_score, 2),
        "skills_match": {
            "matching": list(set(matching_skills)),  # Remove duplicates
            "missing_required": missing_required,
            "missing_preferred": missing_preferred,
            "total_required": len(job_requirements.get('required_skills', [])),
            "total_matching": len(set(matching_skills)),
            "total_preferred": len(job_requirements.get('preferred_skills', []))
        },
        "course_recommendations": course_recommendations,
        "analysis": ai_analysis,
        "job_requirements": job_requirements
    }

# ---------------- ENHANCED CAREER CHATBOT ----------------
def get_career_advice(resume_data, user_message, chat_history):
    # Get recent chat context (last 6 messages for better context)
    recent_context = []
    for msg in chat_history[-6:]:
        role = "User" if msg['role'] == 'user' else "Assistant"
        recent_context.append(f"{role}: {msg['content']}")
    recent_context_str = "\n".join(recent_context)
    
    # Extract resume information for personalization
    personal_info = resume_data.get('personal_info', {})
    skills_data = resume_data.get('skills', {})
    experience = resume_data.get('work_experience', [])
    
    # Flatten skills for better context
    all_skills = []
    for skill_list in skills_data.values():
        if isinstance(skill_list, list):
            all_skills.extend(skill_list)
    
    # Get current role or most recent job
    current_role = "Not specified"
    if experience and len(experience) > 0:
        current_role = experience[0].get('job_title', 'Not specified')
    
    # Get years of experience estimate
    exp_years = len(experience)
    
   
    prompt = f"""
You are **CareerAI**, a friendly and intelligent career mentor who helps users improve their careers, resumes, and job opportunities.

Your personality:
- Supportive and motivating
- Friendly and conversational
- Clear and practical
- Curious about the user's goals

Your goal is to guide the user like a real career coach would.

------------------------------------------------

USER PROFILE
Name: {personal_info.get('full_name', 'User')}
Current Role: {current_role}
Key Skills: {', '.join(all_skills[:8])}
Experience: {exp_years} positions listed
Education: {len(resume_data.get('education', []))} degrees listed

------------------------------------------------

RECENT CONVERSATION CONTEXT
{recent_context_str}

------------------------------------------------

USER MESSAGE
{user_message}

------------------------------------------------

HOW YOU SHOULD RESPOND

1. Speak like a **friendly mentor**, not like a robot.
2. Personalize your advice using the user's **skills, experience, or education**.
3. If the user asks about careers, suggest **roles, skills to learn, or next steps**.
4. If the user asks about jobs, guide them on **improving resume, skills, or preparation**.
5. If the user asks something unclear, ask a **short clarifying question**.
6. Give **practical and actionable advice**.
7. Keep answers **clear and easy to read**.

------------------------------------------------

RESPONSE STYLE

Your response should normally contain:

• A friendly opening  
• Helpful advice or explanation  
• One practical suggestion  
• A follow-up question to continue the conversation  

Use emojis naturally (😊 💡 🚀) but not too many.

Keep responses **3–6 sentences** unless a detailed explanation is necessary.

------------------------------------------------

YOUR RESPONSE:
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Increased for more creative responses
            max_tokens=500     # Increased for longer responses
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in career advice: {e}")
        return "I'm having a little trouble connecting right now. Can you try asking again? 😊"

# ---------------- ROUTES ----------------
@app.route('/')
def landing():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's theme preference
    conn = sqlite3.connect('resume_analyzer.db')
    c = conn.cursor()
    c.execute("SELECT theme_preference FROM users WHERE id = ?", (session['user']['id'],))
    result = c.fetchone()
    conn.close()
    
    theme = result[0] if result else 'light'
    
    return render_template('dashboard.html', 
                         user=session.get('user'),
                         theme=theme)

@app.route('/app')
@login_required
def app_redirect():
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect('resume_analyzer.db')
    c = conn.cursor()
    c.execute("SELECT id, username, full_name, email, password FROM users WHERE username = ? OR email = ?", 
              (username, username))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[4], password):
        session['user'] = {
            'id': user[0],
            'username': user[1],
            'full_name': user[2],
            'email': user[3]
        }
        session.permanent = True
        return jsonify({'success': True, 'user': session['user']})
    
    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    full_name = data.get('fullName')
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    conn = sqlite3.connect('resume_analyzer.db')
    c = conn.cursor()
    
    # Check if user exists
    c.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
    if c.fetchone():
        conn.close()
        return jsonify({'success': False, 'error': 'Username or email already exists'}), 400
    
    # Create new user
    hashed_password = generate_password_hash(password)
    c.execute("INSERT INTO users (username, full_name, email, password) VALUES (?, ?, ?, ?)",
              (username, full_name, email, hashed_password))
    user_id = c.lastrowid
    conn.commit()
    conn.close()
    
    session['user'] = {
        'id': user_id,
        'username': username,
        'full_name': full_name,
        'email': email
    }
    session.permanent = True
    
    return jsonify({'success': True, 'user': session['user']})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))

@app.route('/check-auth')
def check_auth():
    if 'user' in session:
        return jsonify({'authenticated': True, 'user': session['user']})
    return jsonify({'authenticated': False})

@app.route('/toggle-theme', methods=['POST'])
@login_required
def toggle_theme():
    data = request.json
    theme = data.get('theme', 'light')
    
    conn = sqlite3.connect('resume_analyzer.db')
    c = conn.cursor()
    c.execute("UPDATE users SET theme_preference = ? WHERE id = ?", (theme, session['user']['id']))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

# ---------------- MODIFIED UPLOAD ROUTE WITH RESUME VALIDATION ----------------
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    if file_ext not in ['pdf', 'docx', 'txt']:
        return jsonify({'error': 'Invalid file type. Please upload PDF, DOCX, or TXT files only.'}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        text = extract_text(file_path, file_ext)
        char_count = len(text)
        
        # NEW: Validate if the content is actually a resume
        is_valid, validation_message, confidence = is_valid_resume_content(text)
        
        if not is_valid:
            return jsonify({
                'error': validation_message,
                'suggestion': 'Please upload a valid resume file containing work experience, education, and skills sections.',
                'confidence': confidence,
                'text_length': char_count
            }), 400
        
        return jsonify({
            'success': True,
            'text': text,
            'char_count': char_count,
            'filename': filename,
            'validation': {
                'is_valid': True,
                'message': validation_message,
                'confidence': confidence
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    data = request.json
    resume_text = data.get('text', '')
    
    if not resume_text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        sections = extract_resume_sections(resume_text)
        
        # Get resume quality feedback
        feedback = analyze_resume_quality(sections)
        
        return jsonify({
            'success': True,
            'sections': sections,
            'feedback': feedback
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-resume', methods=['POST'])
@login_required
def save_resume():
    data = request.json
    resume_data = data.get('data')
    feedback = data.get('feedback', {})
    user_id = session['user']['id']
    
    try:
        conn = sqlite3.connect('resume_analyzer.db')
        c = conn.cursor()
        c.execute("INSERT INTO resume_data (user_id, data, feedback) VALUES (?, ?, ?)",
                  (user_id, json.dumps(resume_data), json.dumps(feedback)))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-latest-resume', methods=['GET'])
@login_required
def get_latest_resume():
    user_id = session['user']['id']
    
    try:
        conn = sqlite3.connect('resume_analyzer.db')
        c = conn.cursor()
        c.execute("SELECT data, feedback FROM resume_data WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                  (user_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            return jsonify({
                'success': True, 
                'data': json.loads(result[0]),
                'feedback': json.loads(result[1]) if result[1] else {}
            })
        else:
            return jsonify({'success': False, 'error': 'No resume found'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare-job', methods=['POST'])
@login_required
def compare_job():
    data = request.json
    resume_data = data.get('resume_data')
    job_description = data.get('job_description')
    
    if not resume_data or not job_description:
        return jsonify({'error': 'Missing data'}), 400
    
    try:
        analysis = compare_with_job(resume_data, job_description)
        
        # Save to database
        conn = sqlite3.connect('resume_analyzer.db')
        c = conn.cursor()
        c.execute("""INSERT INTO job_comparisons 
                     (user_id, job_description, analysis, match_score) 
                     VALUES (?, ?, ?, ?)""",
                  (session['user']['id'], 
                   job_description[:500], 
                   json.dumps(analysis), 
                   analysis['match_score']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'analysis': analysis})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/career-chat', methods=['POST'])
@login_required
def career_chat():
    data = request.json
    resume_data = data.get('resume_data')
    message = data.get('message')
    chat_history = data.get('chat_history', [])
    
    if not resume_data or not message:
        return jsonify({'error': 'Missing data'}), 400
    
    try:
        advice = get_career_advice(resume_data, message, chat_history)
        
        # Save to database
        conn = sqlite3.connect('resume_analyzer.db')
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (user_id, message, response) VALUES (?, ?, ?)",
                  (session['user']['id'], message, advice))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'response': advice})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-chat-history', methods=['GET'])
@login_required
def get_chat_history():
    user_id = session['user']['id']
    
    try:
        conn = sqlite3.connect('resume_analyzer.db')
        c = conn.cursor()
        c.execute("""SELECT message, response, timestamp FROM chat_history 
                     WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20""", (user_id,))
        results = c.fetchall()
        conn.close()
        
        history = []
        for msg, resp, ts in reversed(results):
            history.append({'role': 'user', 'content': msg})
            history.append({'role': 'assistant', 'content': resp})
        
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
