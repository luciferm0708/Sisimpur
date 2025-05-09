import easyocr
from PIL import Image
import cv2
import re
from pdfminer.high_level import extract_text

def extract_ocr_text_based_pdf(pdf_path):
    text = extract_text(pdf_path)
    print("Extracted text preview:", text[:500])
    # Clean up newlines and spacing
    text = re.sub(r'(?<!\.)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    print("Cleaned text preview:", text[:500])


    # If have any Numbered sentences - Extract numbered sentences
    pattern = re.compile(r'\d+\.\s+.*?(?=\s+\d+\.|$)', re.DOTALL)
    numbered_sentences = pattern.findall(text)

    for sent in numbered_sentences:
        print(sent.strip())

    return text

extract_ocr_text_based_pdf('sisimpur-brain/PDF_Extractor/data/1mb.pdf')


def easy_ocr(image_path):
    reader = easyocr.Reader(['bn', 'en'])
    img = Image.open(image_path)
    result = reader.readtext(img)
    return result

result = easy_ocr('1.jpg')
for bbox, text, prob in result:
    print(f"Detected text: {text}, Confidence: {prob}, Bounding Box: {bbox}")

