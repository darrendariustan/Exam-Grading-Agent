import pdfplumber

def extract_pdf_to_markdown(pdf_path):
    markdown_output = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            tables = page.extract_tables()

            markdown_output += f"\n\n## Page {page_number}\n"

            # 1. Add extracted text (already respects bullets and numbers)
            if text:
                cleaned_text = clean_text_formatting(text)
                markdown_output += cleaned_text

            # 2. Add extracted tables in markdown
            for table in tables:
                markdown_output += "\n\n" + convert_table_to_markdown(table)

    return markdown_output.strip()

def clean_text_formatting(text):
    """
    Cleans extracted text:
    - Standardizes bullet points
    - Ensures consistent line breaks
    """
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")  # Preserve blank lines
            continue
        # Convert bullet characters to markdown-style "-"
        if stripped.startswith(("*", "•", "·", "-")):
            cleaned_lines.append("- " + stripped.lstrip("*•·-").strip())
        # Safely check for numbered lists like "1." or "2-"
        elif len(stripped) > 2 and stripped[:2].isdigit() and stripped[2] in ".-":
            cleaned_lines.append(stripped)
        else:
            cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines) + "\n"


def convert_table_to_markdown(table):
    """
    Converts a list-of-lists table into markdown format.
    """
    header = table[0]
    rows = table[1:]

    # Build the header row
    md_table = "| " + " | ".join(header) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(header)) + " |\n"

    # Add data rows
    for row in rows:
        md_table += "| " + " | ".join(cell if cell else "" for cell in row) + " |\n"

    return md_table

# Example usage
if __name__ == "__main__":
    pdf_path = "../EXAM EVALUATION EXAMPLES (CRA)/CRA_Final_Examen_Rubric.pdf"  # Replace with your PDF path
    markdown = extract_pdf_to_markdown(pdf_path)
    print(markdown)

    # Optionally, write to a markdown file
    with open("output.md", "w") as f:
        f.write(markdown)
