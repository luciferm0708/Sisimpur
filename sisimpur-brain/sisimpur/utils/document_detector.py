"""
Document detector module for Sisimpur Brain.

This module provides functionality to detect document types and languages.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from ..config import MIN_TEXT_LENGTH

logger = logging.getLogger("sisimpur.detector")

def detect_question_paper(text: str, language: str) -> bool:
    """
    Detect if the text is from a question paper.

    Args:
        text: Extracted text from the document
        language: Detected language of the document

    Returns:
        True if the document appears to be a question paper, False otherwise
    """
    # Convert text to lowercase for easier pattern matching
    text_lower = text.lower()

    # Check for common question paper indicators
    if language == "bengali":
        # Bengali question paper patterns
        # Look for question numbers in Bengali (১, ২, ৩, etc.)
        bengali_numbers = re.findall(r'[১২৩৪৫৬৭৮৯০]+\.', text)

        # Look for Bengali MCQ option markers (ক, খ, গ, ঘ)
        bengali_options = re.findall(r'[কখগঘ]\.', text)

        # Look for common Bengali question paper terms
        bengali_terms = [
            "প্রশ্ন", "উত্তর", "পরীক্ষা", "নম্বর", "মোট নম্বর", "সময়", "ঘন্টা", "মিনিট"
        ]

        # Count how many Bengali terms are found
        term_matches = sum(1 for term in bengali_terms if term in text)

        # If we find multiple question numbers or MCQ options, or several question paper terms
        if (len(bengali_numbers) >= 5 or len(bengali_options) >= 10 or term_matches >= 3):
            logger.info(f"Detected Bengali question paper: {len(bengali_numbers)} question numbers, "
                       f"{len(bengali_options)} options, {term_matches} question paper terms")
            return True
    else:
        # English question paper patterns
        # Look for question numbers (1., 2., 3., etc.)
        question_numbers = re.findall(r'\d+\.', text)

        # Look for MCQ option markers (A., B., C., D. or a., b., c., d.)
        mcq_options = re.findall(r'[A-Da-d]\.', text)

        # Look for common English question paper terms
        english_terms = [
            "question", "answer", "exam", "test", "marks", "points", "score",
            "time", "minutes", "hours", "total marks", "section"
        ]

        # Count how many English terms are found
        term_matches = sum(1 for term in english_terms if term in text_lower)

        # If we find multiple question numbers or MCQ options, or several question paper terms
        if (len(question_numbers) >= 5 or len(mcq_options) >= 10 or term_matches >= 3):
            logger.info(f"Detected English question paper: {len(question_numbers)} question numbers, "
                       f"{len(mcq_options)} options, {term_matches} question paper terms")
            return True

    return False

def detect_document_type(file_path: str) -> Dict[str, Any]:
    """
    Detect document type and language from the given file.

    Args:
        file_path: Path to the document file

    Returns:
        Dict with document type, language, and other metadata
    """
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()

    metadata = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_size": file_path.stat().st_size,
        "extension": file_ext,
        "is_question_paper": False,  # Default value, will be updated if detected
    }

    # Detect document type based on extension
    if file_ext in ['.pdf']:
        metadata["doc_type"] = "pdf"
        # Check if PDF is text-based or image-based
        try:
            doc = fitz.open(file_path)
            text_content = ""
            for page in doc:
                text_content += page.get_text()

            if len(text_content.strip()) > MIN_TEXT_LENGTH:  # If substantial text is extracted
                metadata["pdf_type"] = "text_based"

                # Detect language for text-based PDFs
                bengali_chars = sum(1 for c in text_content if '\u0980' <= c <= '\u09FF')
                if bengali_chars > len(text_content) * 0.3:  # If more than 30% Bengali characters
                    metadata["language"] = "bengali"
                else:
                    metadata["language"] = "english"

                # Detect if this is a question paper
                metadata["is_question_paper"] = detect_question_paper(text_content, metadata.get("language", "english"))
            else:
                metadata["pdf_type"] = "image_based"

            # Close the document
            doc.close()
        except Exception as e:
            logger.error(f"Error analyzing PDF: {e}")
            metadata["pdf_type"] = "unknown"

    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        metadata["doc_type"] = "image"

        # Attempt to detect language (Bengali vs English)
        try:
            img = Image.open(file_path)
            # Use pytesseract to detect script
            text = pytesseract.image_to_string(img, lang='ben+eng')

            # Simple heuristic: check for Bengali Unicode range
            bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
            if bengali_chars > len(text) * 0.3:  # If more than 30% Bengali characters
                metadata["language"] = "bengali"
            else:
                metadata["language"] = "english"

            # Detect if this is a question paper
            metadata["is_question_paper"] = detect_question_paper(text, metadata["language"])

        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            metadata["language"] = "unknown"

    else:
        metadata["doc_type"] = "unknown"

    logger.info(f"Detected document type: {metadata}")
    return metadata
