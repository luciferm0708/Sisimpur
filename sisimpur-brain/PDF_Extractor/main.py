import nltk
import re
from pdfminer.high_level import extract_text

pdf_path  = 'sisimpur-brain/PDF_Extractor/data/1mb.pdf'
raw_text = extract_text(pdf_path)
print("Extracted text preview:", raw_text[:500])
# Clean up newlines and spacing
text = re.sub(r'(?<!\.)\n(?!\n)', ' ', raw_text)
text = re.sub(r'\s+', ' ', text)

print("Cleaned text preview:", text[:500])


#If have any Numbered sentences - Extract numbered sentences
pattern = re.compile(r'\d+\.\s+.*?(?=\s+\d+\.|$)', re.DOTALL)
numbered_sentences = pattern.findall(text)

for sent in numbered_sentences:
    print(sent.strip())