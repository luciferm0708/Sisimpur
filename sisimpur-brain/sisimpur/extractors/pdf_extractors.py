"""
PDF extractors for Sisimpur Brain.

This module provides extractors for PDF documents.
"""

import logging
import re
from typing import List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

from .base import BaseExtractor
from ..utils.api_utils import api
from ..config import DEFAULT_GEMINI_MODEL

logger = logging.getLogger("sisimpur.extractors.pdf")

class TextPDFExtractor(BaseExtractor):
    """Extractor for text-based PDF documents"""

    def extract(self, file_path: str) -> str:
        """
        Extract text from text-based PDF.

        Args:
            file_path: Path to the PDF document

        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(file_path)
            text = ""

            for page_num, page in enumerate(doc):
                text += f"--- Page {page_num + 1} ---\n"
                text += page.get_text()
                text += "\n\n"

            doc.close()
            temp_path = self.save_to_temp(text, file_path)
            return text

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

class ImagePDFExtractor(BaseExtractor):
    """Extractor for image-based PDF documents"""

    def extract(self, file_path: str) -> str:
        """
        Extract text from image-based PDF using OCR.

        Args:
            file_path: Path to the PDF document

        Returns:
            Extracted text
        """
        try:
            # Convert PDF to images
            images = convert_from_path(file_path)
            text = ""

            for i, img in enumerate(images):
                text += f"--- Page {i + 1} ---\n"

                # Use Gemini for Bengali, pytesseract for English
                if self.language == "ben":
                    page_text = self._extract_with_gemini(img)
                else:
                    page_text = pytesseract.image_to_string(img, lang=self.language)

                text += page_text
                text += "\n\n"

            temp_path = self.save_to_temp(text, file_path)
            return text

        except Exception as e:
            logger.error(f"Error extracting text from image-based PDF: {e}")
            raise

    def _extract_with_gemini(self, img: Image.Image) -> str:
        """
        Extract text using Gemini for Bengali content.

        Args:
            img: The image to extract text from

        Returns:
            Extracted text
        """
        # First, try to detect if this is a question paper using a simple heuristic
        try:
            # Get a small sample of text to check if it's a question paper
            sample_text = pytesseract.image_to_string(img, lang='ben')
            is_likely_question_paper = self._is_likely_question_paper(sample_text)
        except Exception:
            is_likely_question_paper = False

        # Use a specialized prompt for question papers
        if is_likely_question_paper:
            prompt = (
                "Extract all text from this question paper image with precise formatting. "
                "This appears to be an exam or question paper. "
                "Carefully preserve: "
                "1. Question numbers (including Bengali digits like ১, ২, ৩) "
                "2. All question text with exact wording "
                "3. Multiple choice options (ক, খ, গ, ঘ or A, B, C, D) "
                "4. Any instructions or headers "
                "Maintain original Bengali script and formatting. "
                "Return only the extracted text, no additional comments or analysis."
            )
        else:
            # Standard prompt for general content
            prompt = (
                "Extract all text from this image. Preserve original formatting and language. "
                "If the text is in Bengali, maintain the Bengali script. "
                "Return only the extracted text, no additional comments."
            )

        try:
            response = api.generate_content([prompt, img], model_name=DEFAULT_GEMINI_MODEL)
            return response.text
        except Exception as e:
            logger.error(f"Error using Gemini for OCR: {e}")
            # Fallback to pytesseract
            logger.info("Falling back to pytesseract for OCR")
            return pytesseract.image_to_string(img, lang='ben')

    def _is_likely_question_paper(self, text: str) -> bool:
        """
        Quick check if text is likely from a question paper.

        Args:
            text: Sample text from the image

        Returns:
            True if likely a question paper, False otherwise
        """
        # Check for Bengali question numbers
        bengali_numbers = re.findall(r'[১২৩৪৫৬৭৮৯০]+\.', text)

        # Check for Bengali MCQ options
        bengali_options = re.findall(r'[কখগঘ]\.', text)

        # Check for common Bengali question paper terms
        bengali_terms = ["প্রশ্ন", "উত্তর", "পরীক্ষা", "নম্বর"]
        term_matches = sum(1 for term in bengali_terms if term in text)

        # Check for English question numbers and options too
        question_numbers = re.findall(r'\d+\.', text)
        mcq_options = re.findall(r'[A-Da-d]\.', text)

        # If we find multiple question numbers or MCQ options
        return (len(bengali_numbers) >= 2 or
                len(bengali_options) >= 4 or
                term_matches >= 2 or
                len(question_numbers) >= 2 or
                len(mcq_options) >= 4)
