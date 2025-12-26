"""
Convert BENCHMARK_RESULTS.md to PDF with embedded images
"""
import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import re

def convert_md_to_pdf(md_file: str, output_pdf: str):
    """Convert markdown to PDF with images and tables"""
    
    # Read markdown file
    md_path = Path(md_file)
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Fix image paths to be absolute
    base_dir = md_path.parent
    def fix_image_path(match):
        img_path = match.group(1)
        if not img_path.startswith('http'):
            abs_path = (base_dir / img_path).resolve()
            return f'![Image]({abs_path.as_uri()})'
        return match.group(0)
    
    md_content = re.sub(r'!\[.*?\]\((.*?)\)', fix_image_path, md_content)
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'extra'
        ]
    )
    
    # Add CSS styling for better PDF appearance
    css_style = """
    <style>
        @page {
            size: A4;
            margin: 2cm;
        }
        body {
            font-family: 'Arial', sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            font-size: 24pt;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            page-break-before: always;
        }
        h1:first-of-type {
            page-break-before: avoid;
        }
        h2 {
            color: #34495e;
            font-size: 18pt;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 30px;
        }
        h3 {
            color: #7f8c8d;
            font-size: 14pt;
            margin-top: 20px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 10pt;
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td {
            border: 1px solid #ddd;
            padding: 10px;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            page-break-inside: avoid;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }
        pre code {
            background-color: transparent;
            padding: 0;
        }
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
            font-style: italic;
            color: #555;
            background-color: #f0f7fb;
            padding: 15px 20px;
        }
        strong {
            color: #2c3e50;
        }
        .page-break {
            page-break-after: always;
        }
        hr {
            border: none;
            border-top: 2px solid #bdc3c7;
            margin: 30px 0;
        }
    </style>
    """
    
    # Create full HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Medical RAG Model Comparison</title>
        {css_style}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert to PDF
    print(f"Converting {md_file} to PDF...")
    HTML(string=full_html, base_url=str(base_dir)).write_pdf(
        output_pdf,
        stylesheets=None,
        zoom=1,
        attachments=None
    )
    
    print(f"✅ PDF created successfully: {output_pdf}")
    print(f"📄 File size: {Path(output_pdf).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    # Convert the benchmark results
    md_file = "../BENCHMARK_RESULTS.md"
    output_file = "../BENCHMARK_RESULTS.pdf"
    
    convert_md_to_pdf(md_file, output_file)
