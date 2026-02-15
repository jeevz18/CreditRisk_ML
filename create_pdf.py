"""
Convert SUBMISSION_DOCUMENT.md to PDF
"""
from markdown import markdown
import os

# Read the markdown file
with open('SUBMISSION_DOCUMENT.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Assignment 2 Submission - Credit Risk ML</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        strong {{
            color: #2c3e50;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .checkmark {{
            color: #27ae60;
            font-weight: bold;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
{markdown(md_content, extensions=['tables', 'fenced_code'])}
</body>
</html>
"""

# Save HTML file
with open('SUBMISSION_DOCUMENT.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("[OK] HTML file created: SUBMISSION_DOCUMENT.html")
print("\nTo convert to PDF:")
print("1. Open SUBMISSION_DOCUMENT.html in your browser")
print("2. Press Ctrl+P (Print)")
print("3. Select 'Save as PDF' as destination")
print("4. Save as: 2025AA05523_Assignment2_CreditRisk.pdf")
print("\nAlternatively, use an online converter:")
print("- https://www.html2pdf.com/")
print("- https://www.sejda.com/html-to-pdf")
