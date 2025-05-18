# -*- coding: utf-8 -*-
"""Q&A generator for SISIMPUR Brain (enhanced).

This single file merges your original **qa_generator.py** with the
previous "enhanced" mix‑in so you can drop it straight into your code
base without touching the rest of the pipeline.

Public API compatibility
========================
* **Class name, constructor and return shapes** are unchanged.
* `generate_optimal` still returns *List[Dict[str, str]]*.
* `generate` now accepts an *optional* ``difficulty`` kw‑arg (ignored in
your current prompt templates but ready for future use).  Calls that
omit this parameter keep working because it defaults to *None*.

Key upgrades
------------
* **Log‑scaled quota** – smarter question‑count heuristic using
  `_log_scaled_question_count`.
* **Chunked coverage** – long docs are sliced into overlapping windows
  via `_split_into_chunks`, then merged and deduped.
* **Automatic difficulty inference** – based on Flesch–Kincaid grade via
  *textstat* (optional dependency).  Stubbed if package missing.
* **Robust deduplication** – identical questions removed while
  preserving order.
* **Rich logging** – INFO for user‑level events, DEBUG for internals.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Dict, List, Optional, Sequence

# Optional readability scorer ------------------------------------------------
try:
    import textstat  # type: ignore
except ImportError:  # graceful degradation if textstat not installed
    textstat = None  # type: ignore

from ..utils.api_utils import api
from ..config import (
    QA_GEMINI_MODEL,
    FALLBACK_GEMINI_MODEL,
)

logger = logging.getLogger("sisimpur.generators.qa")


# ---------------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------------

# Note: Some of these utility functions are kept for backward compatibility
# but are no longer used with the new LLM-based approach.

def _estimate_difficulty(text: str) -> str:
    """Return "beginner" | "intermediate" | "advanced" based on readability.

    Note: This function is kept for backward compatibility but is no longer used
    with the new LLM-based approach.
    """
    if textstat is None:
        return "intermediate"
    try:
        fk_grade = textstat.flesch_kincaid_grade(text)
    except Exception:
        return "intermediate"
    if fk_grade < 6:
        return "beginner"
    if fk_grade < 10:
        return "intermediate"
    return "advanced"


def _split_into_chunks(text: str, *, chunk_words: int = 800, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks for processing large documents.

    This function is still used with the new LLM-based approach for large texts.
    """
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks: List[str] = []
    step = chunk_words - overlap
    for start in range(0, len(words), step):
        slice_ = words[start : start + chunk_words]
        if not slice_:
            break
        chunks.append(" ".join(slice_))
    logger.debug("Split text into %d chunks", len(chunks))
    return chunks


def _log_scaled_question_count(total_words: int, *, base: int = 5, cap: int = 100) -> int:
    """Logarithmically grow question count between *base* and *cap*.

    Note: This function is kept for backward compatibility but is no longer used
    with the new LLM-based approach.
    """
    return max(base, min(cap, base + int(3 * math.log10(max(total_words, 1)))))


def _normalise(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _dedup_keep_order(qa_list: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    for item in qa_list:
        key = _normalise(item.get("question", ""))
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
#  Main generator class
# ---------------------------------------------------------------------------
class QAGenerator:
    """Generate question‑answer pairs from extracted text (enhanced)."""

    def __init__(self, language: str = "english"):
        self.language = language
        # Allow callers (e.g., per‑user setting) to set a preferred default
        self.default_difficulty: Optional[str] = None

    # ---------------------------------------------------------------------
    #  Public helpers
    # ---------------------------------------------------------------------
    def generate_optimal(self, text: str, max_questions: Optional[int] = None) -> List[Dict[str, str]]:  # noqa: D401,E501
        """Smart wrapper that uses LLM to generate optimal questions directly."""
        try:
            logger.info("Using LLM to generate optimal questions directly")

            # For very large texts, we still need to chunk
            total_words = len(text.split())

            # Small to medium texts → direct LLM processing
            if total_words < 5000:
                logger.info("Text is small enough for direct processing")
                qa_pairs = self._generate_qa_from_text(text)

                # Apply max_questions limit if specified
                if max_questions is not None and len(qa_pairs) > max_questions:
                    logger.info(f"Limiting to {max_questions} questions as specified")
                    qa_pairs = qa_pairs[:max_questions]

                logger.info(f"LLM generated {len(qa_pairs)} questions")
                return qa_pairs

            # Large texts → chunked generation
            logger.info("Text is large, using chunked processing")
            chunks = _split_into_chunks(text)
            logger.info(f"Split text into {len(chunks)} chunks")

            pooled: List[Dict[str, str]] = []
            for idx, chunk in enumerate(chunks, 1):
                try:
                    logger.info(f"Processing chunk {idx}/{len(chunks)}")
                    partial = self._generate_qa_from_text(chunk)
                    pooled.extend(partial)
                    logger.info(f"Chunk {idx} generated {len(partial)} questions")
                except Exception as err:
                    logger.warning(f"Chunk {idx} failed: {err}")

            # Deduplicate
            deduped = _dedup_keep_order(pooled)

            # Apply max_questions limit if specified
            if max_questions is not None and len(deduped) > max_questions:
                logger.info(f"Limiting to {max_questions} questions as specified")
                deduped = deduped[:max_questions]

            logger.info(f"Final question count after deduplication: {len(deduped)}")
            return deduped

        except Exception as exc:
            logger.error("Error in generate_optimal: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    def generate(self, text: str, num_questions: int = 10, difficulty: Optional[str] = None) -> List[Dict[str, str]]:  # noqa: D401,E501
        """Generate Q&A pairs from text.

        Args:
            text: The text to generate questions from
            num_questions: Maximum number of questions to return (used as a limit)
            difficulty: Not used with the new LLM approach, kept for backward compatibility

        Note: With the new LLM-based approach, the num_questions parameter is used only as a limit
        after generation, not to determine how many questions to generate. The LLM will generate
        the optimal number of questions based on the content.
        """
        try:
            logger.info(f"Generating Q&A pairs (max: {num_questions})")
            qa_pairs = self._generate_qa_from_text(text)

            # Limit the number of questions if needed
            if len(qa_pairs) > num_questions:
                logger.info(f"Limiting from {len(qa_pairs)} to {num_questions} questions")
                qa_pairs = qa_pairs[:num_questions]

            if qa_pairs:
                logger.info(f"Successfully generated {len(qa_pairs)} Q&A pairs")
            else:
                logger.warning("Model returned no Q&A pairs")
            return qa_pairs
        except Exception as err:
            logger.error(f"Error generating Q&A pairs: {err}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    def _generate_qa_from_text(self, text: str) -> List[Dict[str, str]]:
        """Prepare prompt → call Gemini via api → parse JSON."""
        # Use the new LLM-based prompt for generating questions
        prompt_template = """
You are given a passage about an specific context. Your task is to extract the maximum number of valid, askable questions from it, following these strict guidelines:

Accuracy only – Do not invent facts not present in the passage.

Balanced coverage – Cover all major ideas and sections, including biography, career, films, controversies, philanthropy, and public image.

Question types – Generate a mix of:

30% Recall questions (factual)

40% Comprehension/Application questions (cause-effect, comparison, purpose)

30% Higher-order questions (analysis, evaluation, interpretation)

Multiple-choice questions (MCQs) must have:

One correct answer

Three plausible but clearly wrong distractors

No verbatim clues from the passage

Descriptive questions must have thoughtful, text-based answers.

Language – Use clear, neutral English. Avoid slang or idioms.

Output only valid JSON with this schema:

Example:

[
  {{
    "question": "Your question text here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Correct answer here",
    "difficulty": "easy|medium|hard",
    "type": "mcq|descriptive"
  }}
]

Do not include any explanatory notes or extra keys. Output only the JSON array.

Passage:
{text}
"""

        # For Bengali language, we need to adapt the prompt
        if self.language == "bengali":
            prompt_template = """
আপনাকে একটি নির্দিষ্ট প্রসঙ্গ সম্পর্কে একটি অনুচ্ছেদ দেওয়া হয়েছে। আপনার কাজ হল এই কঠোর নির্দেশিকা অনুসরণ করে এটি থেকে সর্বাধিক সংখ্যক বৈধ, জিজ্ঞাসাযোগ্য প্রশ্ন বের করা:

শুধুমাত্র সঠিকতা - অনুচ্ছেদে উপস্থিত নয় এমন তথ্য আবিষ্কার করবেন না।

ভারসাম্যপূর্ণ কভারেজ - সমস্ত প্রধান ধারণা এবং বিভাগ কভার করুন, যার মধ্যে জীবনী, কর্মজীবন, চলচ্চিত্র, বিতর্ক, দাতব্য, এবং জনসাধারণের ভাবমূর্তি অন্তর্ভুক্ত।

প্রশ্নের ধরন - এর মিশ্রণ তৈরি করুন:

30% স্মরণ প্রশ্ন (তথ্যগত)

40% বোধগম্যতা/প্রয়োগ প্রশ্ন (কারণ-প্রভাব, তুলনা, উদ্দেশ্য)

30% উচ্চতর-ক্রম প্রশ্ন (বিশ্লেষণ, মূল্যায়ন, ব্যাখ্যা)

বহুনির্বাচনী প্রশ্ন (MCQ) অবশ্যই থাকতে হবে:

একটি সঠিক উত্তর

তিনটি সম্ভাব্য কিন্তু স্পষ্টতই ভুল বিকল্প

অনুচ্ছেদ থেকে কোন হুবহু সূত্র নেই

বর্ণনামূলক প্রশ্নগুলির অবশ্যই চিন্তাশীল, পাঠ্য-ভিত্তিক উত্তর থাকতে হবে।

ভাষা - স্পষ্ট, নিরপেক্ষ বাংলা ব্যবহার করুন। স্ল্যাং বা বাগধারা এড়িয়ে চলুন।

শুধুমাত্র এই স্কিমা সহ বৈধ JSON আউটপুট করুন:

উদাহরণ:

[
  {{
    "question": "আপনার প্রশ্নের পাঠ্য এখানে",
    "options": ["বিকল্প ক", "বিকল্প খ", "বিকল্প গ", "বিকল্প ঘ"],
    "answer": "সঠিক উত্তর এখানে",
    "difficulty": "সহজ|মাঝারি|কঠিন",
    "type": "বহুনির্বাচনী|বর্ণনামূলক"
  }}
]

কোন ব্যাখ্যামূলক নোট বা অতিরিক্ত কী অন্তর্ভুক্ত করবেন না। শুধুমাত্র JSON অ্যারে আউটপুট করুন।

অনুচ্ছেদ:
{text}
"""

        prompt = prompt_template.format(text=text)

        # -----------------------------------------------------------------
        try:
            logger.debug("Sending prompt to model (%s)", QA_GEMINI_MODEL)
            try:
                response = api.generate_content(prompt, model_name=QA_GEMINI_MODEL)
            except Exception as primary_err:
                logger.warning("Primary model failed, using fallback: %s", primary_err)
                response = api.generate_content(prompt, model_name=FALLBACK_GEMINI_MODEL)

            json_str = response.text
            # Extract JSON from triple‑backtick fences if present --------
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```", 1)[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```", 1)[0].strip()

            try:
                qa_data = json.loads(json_str)

                # Handle the new JSON format which is a direct array
                if isinstance(qa_data, list):
                    logger.info(f"Successfully extracted {len(qa_data)} questions using new format")
                    return qa_data

                # Handle the old format for backward compatibility
                elif isinstance(qa_data, dict) and isinstance(qa_data.get("questions"), list):
                    logger.info(f"Successfully extracted {len(qa_data['questions'])} questions using old format")
                    return qa_data["questions"]

                logger.warning("Model returned unexpected JSON format: %s", qa_data)
                return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {json_str}")
                return []

        except Exception as err:
            logger.error("Model call/parsing failed: %s", err, exc_info=True)
            return []
