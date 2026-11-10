
import pandas as pd

# 1. Load dataset
df = pd.read_csv(
    "jobs_final_5_columns.csv",
    encoding="utf-8-sig"
)


# 2. Job classification function
def classify_role(title):

    t = str(title).lower()

    # Non-technical business roles
    non_technical = [
        "account manager",
        "sales specialist",
        "sales engineer",
        "talent acquisition",
        "communications intern",
        "partner operations",
        "global alliances",
        "renewals",
        "customer support",
        "channel account",
        "risk analyst",
        "analyst relations"
    ]

    if any(x in t for x in non_technical):
        return "Non-Technical / Business"

    # AI and Machine Learning
    if any(x in t for x in [
        "artificial intelligence",
        "machine learning",
        "conversational ai",
        "applied ai",
        "ai engineer",
        "ai experience",
        "ml engineer",
        "computer vision",
        "natural language processing",
        "nlp"
    ]):
        return "AI / Machine Learning"

    # Data Engineering and Data Science
    if any(x in t for x in [
        "data engineer",
        "data scientist",
        "data analyst",
        "analytics engineer",
        "big data"
    ]):
        return "Data Engineering / Science"

    # Full-Stack Development
    if any(x in t for x in [
        "full-stack",
        "full stack",
        "fullstack"
    ]):
        return "Full-Stack Development"

    # Mobile Development
    if any(x in t for x in [
        "mobile developer",
        "mobile engineer",
        "android developer",
        "ios developer",
        "ios software",
        "flutter",
        "react native"
    ]):
        return "Mobile Development"

    # Frontend Development
    if any(x in t for x in [
        "frontend",
        "front-end",
        "front end",
        "web engineer",
        "web developer",
        "web development",
        "web design",
        "web implementation",
        "ui engineer",
        "ui development",
        "react engineer"
    ]):
        return "Frontend / Web Development"

    # Backend Development
    if any(x in t for x in [
        "backend",
        "back-end",
        "back end",
        "php engineer",
        "web services",
        "apigee",
        "mulesoft"
    ]):
        return "Backend Development"

    # QA and Testing
    if any(x in t for x in [
        "sdet",
        "software development engineer in test",
        "quality assurance",
        "qa tester",
        "test engineer",
        "software tester"
    ]):
        return "QA / Testing"

    # Cybersecurity
    if any(x in t for x in [
        "cybersecurity",
        "security engineer",
        "information security",
        "platform security",
        "software security",
        "detection & response"
    ]):
        return "Cybersecurity"

    # DevOps, Cloud and Networks
    if any(x in t for x in [
        "devops",
        "cloudops",
        "cloud engineer",
        "cloud architect",
        "cloud solutions architect",
        "infrastructure",
        "site reliability",
        " sre ",
        "network engineer",
        "kubernetes",
        "platform engineer",
        "linux administrator"
    ]):
        return "DevOps / Cloud / Networks"

    # IT Systems and Support
    if any(x in t for x in [
        "systems engineer",
        "system administrator",
        "salesforce administrator",
        "business systems",
        "help desk",
        "service desk",
        "it support",
        "support engineer",
        "enterprise support",
        "web support",
        "consumer support",
        "customer solutions"
    ]):
        return "IT Systems / Support"

    # Software Architecture and Leadership
    if any(x in t for x in [
        "solutions architect",
        "software architect",
        "head of engineering",
        "development manager",
        "senior staff engineer",
        "software director",
        "engineering manager"
    ]):
        return "Software Architecture / Leadership"

    # Software Engineering
    if any(x in t for x in [
        "software",
        "developer",
        "programmer",
        "programming",
        "software consultant",
        ".net",
        "c++",
        "golang",
        "embedded",
        "python engineer",
        "web scraping",
        "web scraper",
        "scraping engineer",
        "scrapy",
        "engineer"
    ]):
        return "Software Engineering"

    # Other / Manual Review
    return "Other / Review"


# 3. Apply classification
df["Job_Role"] = df["Job_Title"].apply(classify_role)


# 4. Display results
print("=" * 55)
print("FINAL JOB ROLE CLASSIFICATION")
print("=" * 55)

print("\nTotal jobs:", len(df))

print("\nJobs by role:")
print(df["Job_Role"].value_counts())

print("\nPercentage by role:")
print(
    (df["Job_Role"].value_counts(normalize=True) * 100)
    .round(2)
)


# 5. Save classified dataset
df.to_csv(
    "jobs_with_roles.csv",
    index=False,
    encoding="utf-8-sig"
)

# 6. Save summary
df["Job_Role"].value_counts().to_csv(
    "job_role_summary.csv",
    header=["Number_of_Jobs"],
    encoding="utf-8-sig"
)

print("\nFiles saved successfully:")
print("- jobs_with_roles.csv")
print("- job_role_summary.csv")
