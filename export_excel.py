
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# Read final dataset
df = pd.read_csv("jobs_final_5_columns.csv", encoding="utf-8-sig")

# Keep the five main columns
columns = [
    "Job_Title",
    "Location",
    "Categories",
    "Experience_Level",
    "Required_Skills"
]

df = df[columns].fillna("Not Specified")

# Separate skills: one skill per row in another sheet
skills_data = []

for job_number, row in df.iterrows():
    skills = str(row["Required_Skills"]).split(",")

    for skill in skills:
        skill = skill.strip()

        if skill:
            skills_data.append({
                "Job_Number": job_number + 1,
                "Job_Title": row["Job_Title"],
                "Required_Skill": skill
            })

skills_df = pd.DataFrame(skills_data)

# Export to Excel
with pd.ExcelWriter("jobs_final_organized.xlsx", engine="openpyxl") as writer:

    df.to_excel(
        writer,
        sheet_name="Jobs",
        index=False
    )

    skills_df.to_excel(
        writer,
        sheet_name="Skills",
        index=False
    )

    for ws in writer.book.worksheets:

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        # Format header
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(
                start_color="580F41",
                end_color="580F41",
                fill_type="solid"
            )
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center"
            )

        # Adjust column widths
        for column in ws.columns:
            max_length = max(
                len(str(cell.value or ""))
                for cell in column
            )

            ws.column_dimensions[
                get_column_letter(column[0].column)
            ].width = min(max_length + 3, 50)

print("Excel file created successfully!")
print("Total jobs:", len(df))
print("Total skill records:", len(skills_df))
