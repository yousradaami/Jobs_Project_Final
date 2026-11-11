
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

print("=" * 65)
print("       ORGANIZING CLEANED JOBS DATASET")
print("=" * 65)

# =====================================================
# 1. FILE PATHS
# =====================================================

input_file = r"C:\Users\pc\Desktop\Jobs_Scraping_Analysis\jobs_cleaned_full.csv"

output_file = "jobs_cleaned_organized.xlsx"

# =====================================================
# 2. READ CLEANED CSV DATASET
# =====================================================

try:

    print("\nLoading cleaned dataset...")

    df = pd.read_csv(
        input_file,
        encoding="utf-8-sig",
        sep=",",
        quotechar='"',
        engine="python",
        dtype=str,
        keep_default_na=False
    )

    # =================================================
    # 3. DATASET INFORMATION
    # =================================================

    print("\nDATASET INFORMATION")
    print("-" * 50)

    print("Total jobs:", len(df))
    print("Total columns:", len(df.columns))

    print("\nColumn names:")

    for column in df.columns:
        print("-", column)

    # =================================================
    # 4. EXPORT TO EXCEL
    # =================================================

    print("\nCreating organized Excel file...")

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Cleaned_Jobs",
            index=False
        )

        ws = writer.sheets["Cleaned_Jobs"]

        # Header formatting
        for cell in ws[1]:

            cell.fill = PatternFill(
                fill_type="solid",
                fgColor="580F41"
            )

            cell.font = Font(
                bold=True,
                color="FFFFFF",
                size=11
            )

            cell.alignment = Alignment(
                horizontal="center",
                vertical="center"
            )

        ws.row_dimensions[1].height = 30

        # Column widths and cell formatting
        for col_num, column in enumerate(df.columns, 1):

            letter = get_column_letter(col_num)

            if any(
                word in column.lower()
                for word in ["description", "skills", "evidence"]
            ):
                width = 45
            else:
                width = min(
                    max(len(str(column)) + 5, 15),
                    35
                )

            ws.column_dimensions[letter].width = width

            for row in range(2, len(df) + 2):

                cell = ws.cell(
                    row=row,
                    column=col_num
                )

                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True
                )

        # Filters and frozen header
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

    # =================================================
    # 5. FINAL VERIFICATION
    # =================================================

    check_df = pd.read_excel(
        output_file,
        sheet_name="Cleaned_Jobs",
        dtype=str
    )

    print("\n" + "=" * 65)
    print("       EXCEL EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    print("\nOriginal CSV rows:", len(df))
    print("Excel rows:", len(check_df))

    print("Original columns:", len(df.columns))
    print("Excel columns:", len(check_df.columns))

    if len(df) == len(check_df) and len(df.columns) == len(check_df.columns):

        print("\nVERIFICATION: PASSED")

    else:

        print("\nWARNING: DATASET SIZE MISMATCH")

    print("\nOutput file:", output_file)
    print("Original CSV was not modified.")

    print("\nDONE!")

except Exception as e:

    print("\nERROR OCCURRED:")
    print(e)

