"""
Convert BENCHMARK_RESULTS.md to standalone HTML with embedded images
"""
import markdown
from pathlib import Path
import re
import base64
import mimetypes

def embed_image(img_path):
    """Convert image to base64 data URL"""
    try:
        with open(img_path, 'rb') as img_file:
            img_data = img_file.read()
            mime_type = mimetypes.guess_type(img_path)[0] or 'image/png'
            b64_data = base64.b64encode(img_data).decode('utf-8')
            return f'data:{mime_type};base64,{b64_data}'
    except Exception as e:
        print(f"Warning: Could not embed {img_path}: {e}")
        return img_path

def convert_latex_table_to_html(latex_table):
    """Convert LaTeX table to HTML table"""
    try:
        # Extract table content
        lines = latex_table.strip().split('\n')
        html_rows = []
        in_header = True
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('\\') or line.startswith('%'):
                if '\\midrule' in line:
                    in_header = False
                continue
            
            # Parse table row
            if '&' in line:
                cells = [cell.strip().replace('\\textbf{', '').replace('}', '').replace('$\\uparrow$', '↑').replace('$\\downarrow$', '↓') 
                        for cell in line.split('&')]
                cells[-1] = cells[-1].replace('\\\\', '').strip()
                
                if in_header:
                    html_rows.append('<tr>' + ''.join(f'<th>{cell}</th>' for cell in cells) + '</tr>')
                else:
                    # Bold the best values
                    row_html = '<tr>'
                    for cell in cells:
                        if cell.strip():
                            row_html += f'<td>{cell}</td>'
                    row_html += '</tr>'
                    html_rows.append(row_html)
        
        html_table = f"""
<table class="latex-table">
<thead>
{html_rows[0] if html_rows else ''}
</thead>
<tbody>
{''.join(html_rows[1:])}
</tbody>
</table>
"""
        return html_table
    except Exception as e:
        print(f"Warning: Could not convert LaTeX table: {e}")
        return f'<pre>{latex_table}</pre>'

def convert_md_to_html(md_file: str, output_html: str):
    """Convert markdown to standalone HTML with embedded images"""
    
    # Read markdown file
    md_path = Path(md_file)
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    base_dir = md_path.parent
    
    # Convert LaTeX tables to HTML tables
    def replace_latex_table(match):
        latex_table = match.group(0)
        return convert_latex_table_to_html(latex_table)
    
    md_content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', replace_latex_table, md_content, flags=re.DOTALL)
    
    # Find all images and embed them
    def embed_image_in_md(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        
        if not img_path.startswith('http'):
            full_path = (base_dir / img_path).resolve()
            if full_path.exists():
                print(f"Embedding image: {img_path}")
                data_url = embed_image(full_path)
                return f'![{alt_text}]({data_url})'
            else:
                print(f"Warning: Image not found: {full_path}")
        
        return match.group(0)
    
    # Embed all images as base64
    md_content = re.sub(r'!\[(.*?)\]\((.*?)\)', embed_image_in_md, md_content)
    
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
    
    # Professional CSS styling
    css_style = """
    <style>
        * {
            box-sizing: border-box;
        }
        
        @media print {
            @page {
                size: A4;
                margin: 2cm;
            }
            
            h1, h2, h3 {
                page-break-after: avoid;
            }
            
            img, table, pre {
                page-break-inside: avoid;
            }
            
            h1 {
                page-break-before: always;
            }
            
            h1:first-of-type {
                page-break-before: avoid;
            }
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #fff;
        }
        
        h1 {
            color: #1a1a1a;
            font-size: 28pt;
            font-weight: 700;
            border-bottom: 4px solid #0066cc;
            padding-bottom: 12px;
            margin-top: 50px;
            margin-bottom: 25px;
        }
        
        h1:first-of-type {
            margin-top: 0;
        }
        
        h2 {
            color: #2c3e50;
            font-size: 20pt;
            font-weight: 600;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 10px;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        
        h3 {
            color: #34495e;
            font-size: 16pt;
            font-weight: 600;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        
        h4 {
            color: #555;
            font-size: 13pt;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        p {
            margin: 12px 0;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 25px auto;
            font-size: 10pt;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        table.latex-table {
            max-width: 95%;
            font-size: 9.5pt;
        }
        
        th {
            background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
            color: white;
            padding: 14px 12px;
            text-align: center;
            font-weight: 600;
            border: 1px solid #0052a3;
        }
        
        td {
            border: 1px solid #ddd;
            padding: 12px;
            vertical-align: middle;
            text-align: center;
        }
        
        td:first-child {
            text-align: left;
            font-weight: 600;
        }
        width: 90%;
            height: auto;
            display: block;
            margin: 30px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        @media print {
            img {
                width: 85%;
                max-height: 400px;
                object-fit: contain;
            }
            background-color: #e8f4f8;
        }
        
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 30px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        code {
            background-color: #f4f4f4;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 9.5pt;
            color: #c7254e;
        }
        
        pre {
            background-color: #f6f8fa;
            padding: 20px;
            border-radius: 6px;
            overflow-x: auto;
            border-left: 4px solid #0066cc;
            margin: 20px 0;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
            color: #24292e;
            font-size: 10pt;
        }
        
        blockquote {
            border-left: 5px solid #0066cc;
            padding: 15px 25px;
            margin: 25px 0;
            font-style: italic;
            color: #555;
            background: linear-gradient(to right, #e8f4f8 0%, #ffffff 100%);
            border-radius: 0 4px 4px 0;
        }
        
        blockquote p {
            margin: 8px 0;
        }
        
        strong {
            color: #1a1a1a;
            font-weight: 600;
        }
        
        em {
            color: #555;
        }
        
        hr {
            border: none;
            border-top: 2px solid #bdc3c7;
            margin: 40px 0;
        }
        
        ul, ol {
            margin: 15px 0;
            padding-left: 30px;
        }
        
        li {
            margin: 8px 0;
        }
        
        a {
            color: #0066cc;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .header-info {
            text-align: center;
            color: #666;
            font-size: 10pt;
            margin-bottom: 30px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }
    </style>
    """
    
    # Create full HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research-Grade Model Comparison for Medical RAG Systems</title>
    {css_style}
</head>
<body>
    {html_content}
    <div class="header-info" style="margin-top: 50px;">
        <p>Generated on December 26, 2025 | Medical RAG Benchmark Results</p>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"✅ HTML created successfully: {output_html}")
    print(f"📄 File size: {Path(output_html).stat().st_size / 1024:.1f} KB")
    print(f"\n📋 Next steps:")
    print(f"   1. Open {output_html} in your browser")
    print(f"   2. Press Ctrl+P to print")
    print(f"   3. Select 'Save as PDF'")
    print(f"   4. Save as BENCHMARK_RESULTS.pdf")


if __name__ == "__main__":
    md_file = "BENCHMARK_RESULTS.md"
    output_file = "BENCHMARK_RESULTS.html"
    
    convert_md_to_html(md_file, output_file)
