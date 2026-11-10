
import requests
import pandas as pd
import time
import re
import html
import xml.etree.ElementTree as ET

from datetime import datetime, timezone

# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

TARGET = 2000
MAX_PAGES_PER_KEYWORD = 30
DELAY = 1.2

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python Job Research"
})

all_jobs = {}
source_errors = []

# ==========================================
# 2. IT SEARCH KEYWORDS
# ==========================================

SEARCH_KEYWORDS = [
    "software developer",
    "software engineer",
    "web developer",
    "frontend developer",
    "backend developer",
    "full stack developer",
    "Python developer",
    "Java developer",
    "JavaScript developer",
    "React developer",
    "PHP developer",
    "mobile developer",
    "Android developer",
    "iOS developer",
    "data analyst",
    "data scientist",
    "machine learning engineer",
    "AI engineer",
    "cybersecurity",
    "security engineer",
    "DevOps engineer",
    "cloud engineer",
    "network engineer",
    "system administrator",
    "database administrator",
    "QA engineer",
    "IT support",
    "technical support",
    "site reliability engineer",
    "embedded software engineer"
]

IT_TITLE_PATTERN = re.compile(
    r"\b(software|developer|programmer|programming|"
    r"frontend|front-end|backend|back-end|full.?stack|"
    r"mobile developer|android|ios developer|"
    r"data analyst|data scientist|machine learning|"
    r"cybersecurity|cyber security|security engineer|"
    r"devops|cloud engineer|network engineer|"
    r"system administrator|sysadmin|database administrator|"
    r"database engineer|dba|qa engineer|quality assurance|"
    r"test automation|it support|technical support|help desk|"
    r"site reliability|sre|embedded software|firmware|"
    r"software architect|solutions architect)\b",
    re.I
)

TECH_PATTERN = re.compile(
    r"\b(python|java|javascript|typescript|sql|php|"
    r"react|angular|vue|linux|docker|kubernetes|aws|azure|gcp|"
    r"machine learning|artificial intelligence|cybersecurity|"
    r"networking|database|programming|software|api|devops)\b",
    re.I
)

# ==========================================
# 3. SKILLS EXTRACTION
# ==========================================

SKILLS_LIST = [
    "Python", "Java", "JavaScript", "TypeScript",
    "C++", "C#", "PHP", "SQL", "R",
    "React", "Angular", "Vue.js", "Node.js",
    "HTML", "CSS", "Django", "Flask",
    "Spring", "Laravel", ".NET",
    "MongoDB", "MySQL", "PostgreSQL",
    "Redis", "Firebase",
    "Linux", "Windows Server",
    "Docker", "Kubernetes", "Jenkins",
    "AWS", "Azure", "Google Cloud",
    "Git", "GitHub", "GitLab",
    "REST API", "GraphQL",
    "Machine Learning", "Deep Learning",
    "Artificial Intelligence", "TensorFlow",
    "PyTorch", "Pandas", "NumPy",
    "Data Analysis", "Power BI", "Tableau",
    "Excel", "Cybersecurity",
    "Networking", "TCP/IP",
    "DevOps", "CI/CD", "Agile",
    "Scrum", "Selenium", "Jira",
    "Figma", "Kotlin", "Swift",
    "Flutter", "React Native",
    "Bash", "Terraform", "Ansible"
]

def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, list):
        value = ", ".join(map(str, value))

    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()

def extract_skills(description, title):
    text = clean_text(description) + " " + clean_text(title)

    found = []

    for skill in SKILLS_LIST:
        pattern = (
            r"(?<![A-Za-z0-9])"
            + re.escape(skill)
            + r"(?![A-Za-z0-9])"
        )

        if re.search(pattern, text, re.I):
            found.append(skill)

    return ", ".join(found) if found else "Not specified"

# ==========================================
# 4. CATEGORY CLASSIFICATION
# ==========================================

def extract_category(title, categories="", description=""):

    title_text = clean_text(title).lower()
    category_text = clean_text(categories).lower()
    combined = title_text + " " + category_text

    if re.search(r"cybersecurity|cyber security|security engineer", combined):
        return "Cybersecurity"

    if re.search(r"data analyst|data scientist|data engineer|business intelligence", combined):
        return "Data Analysis"

    if re.search(r"machine learning|artificial intelligence|\bai engineer\b", combined):
        return "Artificial Intelligence"

    if re.search(r"frontend|front-end|backend|back-end|web developer|full.?stack", combined):
        return "Web Development"

    if re.search(r"android|ios developer|mobile developer|flutter|react native", combined):
        return "Mobile Development"

    if re.search(r"devops|cloud engineer|site reliability|\bsre\b", combined):
        return "Cloud & DevOps"

    if re.search(r"network engineer|network administrator|networking", combined):
        return "Networking"

    if re.search(r"database administrator|database engineer|\bdba\b", combined):
        return "Database Administration"

    if re.search(r"\bqa\b|quality assurance|test automation|software tester", combined):
        return "Software Testing"

    if re.search(r"it support|technical support|help desk|system administrator|sysadmin", combined):
        return "IT Support & Systems"

    if re.search(r"software|programmer|developer|programming|embedded|firmware", combined):
        return "Software Development"

    return "Other IT"

# ==========================================
# 5. IT JOB VALIDATION
# ==========================================

def is_it_job(title, categories="", description=""):

    title = clean_text(title)
    categories = clean_text(categories)
    description = clean_text(description)

    if IT_TITLE_PATTERN.search(title):
        return True

    if re.search(
        r"\b(software development|information technology|"
        r"software engineering|data science|devops|"
        r"cybersecurity)\b",
        categories,
        re.I
    ):
        return True

    if re.search(r"\b(engineer|analyst|specialist)\b", title, re.I):
        return bool(
            TECH_PATTERN.search(title + " " + description[:2000])
        )

    return False

# ==========================================
# 6. CREATE JOB RECORD
# ==========================================

def make_record(
    source, title, company, location,
    date, description, url, apply_url="",
    categories="", keyword="", external_id=""
):

    title = clean_text(title)
    description = clean_text(description)
    company = clean_text(company)
    location = clean_text(location)
    categories = clean_text(categories)

    if not title or not url:
        return None

    if not is_it_job(title, categories, description):
        return None

    if not location:
        location = "Not specified"

    skills = extract_skills(description, title)
    category = extract_category(title, categories, description)

    key = "|".join([
        title.lower(),
        company.lower(),
        location.lower()
    ])

    return {
        "Job_Title": title,
        "Skills": skills,
        "Location": location,
        "Category": category,
        "_key": key
    }

def add_job(record):

    if record is None:
        return

    key = record.pop("_key")

    if key not in all_jobs and len(all_jobs) < TARGET:
        all_jobs[key] = record

# ==========================================
# 7. HTTP REQUEST
# ==========================================

def request_get(url, params=None):

    for attempt in range(4):

        try:
            response = session.get(
                url,
                params=params,
                timeout=(15, 90)
            )

            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Waiting {wait} seconds...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response

        except requests.RequestException:

            if attempt == 3:
                raise

            time.sleep(3 * (attempt + 1))

    return None

# ==========================================
# 8. HIMALAYAS API
# ==========================================

def scrape_himalayas():

    print("\n========== HIMALAYAS ==========")

    url = "https://himalayas.app/jobs/api/search"

    for keyword in SEARCH_KEYWORDS:

        if len(all_jobs) >= TARGET:
            break

        print(f"\nSearching: {keyword}")

        for page in range(1, MAX_PAGES_PER_KEYWORD + 1):

            if len(all_jobs) >= TARGET:
                break

            try:
                response = request_get(
                    url,
                    params={
                        "q": keyword,
                        "page": page
                    }
                )

                data = response.json()
                jobs = data.get("jobs", [])

                if not jobs:
                    break

                for job in jobs:

                    if len(all_jobs) >= TARGET:
                        break

                    expiry = job.get("expiryDate")

                    if expiry:
                        if float(expiry) < datetime.now(
                            timezone.utc
                        ).timestamp():
                            continue

                    guid = str(job.get("guid", ""))

                    if guid.startswith("/"):
                        job_url = "https://himalayas.app" + guid
                    elif guid.startswith("http"):
                        job_url = guid
                    else:
                        job_url = "https://himalayas.app/jobs"

                    pub_date = job.get("pubDate")

                    if isinstance(pub_date, (int, float)):
                        pub_date = datetime.fromtimestamp(
                            pub_date, timezone.utc
                        ).isoformat()

                    categories = (
                        (job.get("categories") or [])
                        + (job.get("parentCategories") or [])
                    )

                    record = make_record(
                        source="Himalayas",
                        title=job.get("title"),
                        company=job.get("companyName"),
                        location=job.get("locationRestrictions"),
                        date=pub_date,
                        description=job.get("description") or job.get("excerpt"),
                        url=job_url,
                        apply_url=job.get("applicationLink", ""),
                        categories=categories,
                        keyword=keyword,
                        external_id=guid
                    )

                    add_job(record)

                print(f"Page {page} | Unique IT jobs: {len(all_jobs)}")
                time.sleep(DELAY)

            except Exception as error:
                print("Himalayas error:", error)
                source_errors.append(f"Himalayas - {keyword}: {error}")
                break

# ==========================================
# 9. REMOTE OK API
# ==========================================

def scrape_remoteok():

    if len(all_jobs) >= TARGET:
        return

    print("\n========== REMOTE OK ==========")

    url = "https://remoteok.com/api"

    try:
        response = request_get(url)
        data = response.json()

        for job in data:

            if len(all_jobs) >= TARGET:
                break

            if not isinstance(job, dict) or not job.get("id"):
                continue

            tags = job.get("tags", [])

            record = make_record(
                source="Remote OK",
                title=job.get("position"),
                company=job.get("company"),
                location=job.get("location"),
                date=job.get("date"),
                description=job.get("description"),
                url=job.get("url"),
                apply_url=job.get("apply_url", ""),
                categories=tags,
                keyword="Remote OK JSON",
                external_id=job.get("id")
            )

            add_job(record)

        print("Remote OK collection finished.")

    except Exception as error:
        print("Remote OK error:", error)
        source_errors.append(f"Remote OK: {error}")

# ==========================================
# 10. WE WORK REMOTELY RSS
# ==========================================

def scrape_weworkremotely():

    if len(all_jobs) >= TARGET:
        return

    print("\n========== WE WORK REMOTELY ==========")

    feeds = {
        "Programming":
            "https://weworkremotely.com/categories/remote-programming-jobs.rss",

        "DevOps":
            "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss"
    }

    for category, url in feeds.items():

        if len(all_jobs) >= TARGET:
            break

        try:
            response = request_get(url)
            root = ET.fromstring(response.content)
            items = root.findall(".//item")

            for item in items:

                if len(all_jobs) >= TARGET:
                    break

                title = item.findtext("title", "")
                link = item.findtext("link", "")
                description = item.findtext("description", "")
                pub_date = item.findtext("pubDate", "")

                parts = title.split(":", 1)

                if len(parts) == 2:
                    company = parts[0].strip()
                    job_title = parts[1].strip()
                else:
                    company = ""
                    job_title = title

                record = make_record(
                    source="We Work Remotely",
                    title=job_title,
                    company=company,
                    location="Remote",
                    date=pub_date,
                    description=description,
                    url=link,
                    categories=category,
                    keyword="RSS " + category
                )

                add_job(record)

            print(category, "feed collected.")

        except Exception as error:
            print("We Work Remotely error:", error)
            source_errors.append(
                f"We Work Remotely {category}: {error}"
            )

# ==========================================
# 11. REMOTIVE API
# ==========================================

def scrape_remotive():

    if len(all_jobs) >= TARGET:
        return

    print("\n========== REMOTIVE ==========")

    url = "https://remotive.com/api/remote-jobs"

    try:
        response = request_get(
            url,
            params={"category": "software-dev"}
        )

        data = response.json()
        jobs = data.get("jobs", [])

        print("Remotive jobs received:", len(jobs))

        for job in jobs:

            if len(all_jobs) >= TARGET:
                break

            record = make_record(
                source="Remotive",
                title=job.get("title"),
                company=job.get("company_name"),
                location=job.get("candidate_required_location"),
                date=job.get("publication_date"),
                description=job.get("description"),
                url=job.get("url"),
                apply_url=job.get("url"),
                categories=job.get("category"),
                keyword="Software Development",
                external_id=job.get("id")
            )

            add_job(record)

    except Exception as error:
        print("Remotive error:", error)
        source_errors.append(f"Remotive: {error}")

# ==========================================
# 12. SAVE FINAL DATASET
# ==========================================

def save_results():

    print("\n========== FINAL RESULTS ==========")

    df = pd.DataFrame(list(all_jobs.values()))

    if df.empty:
        print("No jobs were collected.")
        return

    df.drop_duplicates(
        subset=["Job_Title", "Location"],
        inplace=True
    )

    # Keep ONLY the four requested columns
    df = df[
        ["Job_Title", "Skills", "Location", "Category"]
    ]

    # Replace empty values
    for column in df.columns:
        df[column] = df[column].replace(
            r"^\s*$", "Not specified", regex=True
        ).fillna("Not specified")

    df.to_csv(
        "jobs_informatique.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTotal unique IT jobs:", len(df))
    print("Target:", TARGET)
    print("Remaining:", max(0, TARGET - len(df)))

    print("\nDataset columns:")
    print(list(df.columns))

    print("\nJobs by category:")
    print(df["Category"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDataset saved: jobs_informatique.csv")

    if len(df) >= TARGET:
        print("\nTARGET REACHED: 2000 JOBS!")
    else:
        print("\nTarget not reached yet.")

    if source_errors:
        print("\nSource errors:")
        for error in source_errors:
            print("-", error)

# ==========================================
# 13. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("STARTING IT JOB DATA COLLECTION")
    print("Target:", TARGET)

    scrape_weworkremotely()
    scrape_remoteok()
    scrape_remotive()
    scrape_himalayas()

    save_results()

    print("\nCOLLECTION FINISHED.")