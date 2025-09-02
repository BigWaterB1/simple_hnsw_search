# chunk.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import PyPDF2

def read_data() -> str:
    """Read content from data.md file"""
    with open("data.md", "r", encoding="utf-8") as f:
        return f.read()

def convert_pdf_to_md(pdf_path: str) -> str:
    """Convert PDF file to markdown text"""
    md_content = ""
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                md_content += f"{text}\n\n"
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")
    return md_content

def get_all_content(data_dir: str) -> list[str]:
    """Get content from all files as separate strings to avoid mixing content"""
    file_contents = []
    
    # Process files in data folder
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(data_dir, filename)
                print(f"Converting PDF: {filename}")
                md_text = convert_pdf_to_md(pdf_path)
                file_contents.append(f"# {filename}\n\n{md_text}")
            elif filename.endswith('.md'):
                md_path = os.path.join(data_dir, filename)
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        file_contents.append(f"# {filename}\n\n{f.read()}")
                except Exception as e:
                    print(f"Error reading {md_path}: {e}")
    
    return file_contents

def get_chunks(data_dir: str = "data") -> list[str]:
    """Get chunks from all available content, ensuring different files aren't split into same chunk"""
    file_contents = get_all_content(data_dir)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    
    all_chunks = []
    for content in file_contents:
        chunks = text_splitter.split_text(content)
        all_chunks.extend(chunks)
    
    return all_chunks

if __name__ == '__main__':
    chunks = get_chunks()
    for c in chunks:
        print(c)
        print("--------------")