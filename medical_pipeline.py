"""
Medical Prescription AI Pipeline (Split-Architecture)
=====================================================
Optimized for Apple Silicon (M3 Ultra - 96GB VRAM)

This script processes multiple PDF medical records, extracting handwritten and typed 
data using a Vision-Language Model, structures them chronologically across all documents, 
and launches an interactive Q&A session using a heavy Text LLM.

Key Features:
- Multi-PDF Support: Process entire directories or lists of patient records at once.
- 300 DPI PDF-to-PNG Extraction: Preserves complex handwriting details.
- Strict JSON VLM Extraction: Forces the vision model to output structured data.
- Date Interpolation: Infers missing prescription dates based on surrounding pages.
- Chronological Sorting: Reorders out-of-sequence PDF pages into a true timeline.
- Interactive Medical REPL: Chat with your data using a massive context window.

=====================================================

python medical_pipeline.py \
  --pdfs *.pdf \
  --model qwen3-vl \
  --main-urls http://127.0.0.1:8033/v1/chat/completions:3 \
  --synthesis-model qwen3.5-35b \
  --synthesis-url http://127.0.0.1:8034/v1/chat/completions

./llama-server \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:Q6_K \
  --host 0.0.0.0 --port 8034 \
  -ngl 999 -fa on \
  -c 80000 \
  -b 1024 \
  -ub 1024 \
  -np 1
"""

import os
import base64
import requests
import glob
import time
import hashlib
import logging
import argparse
import shutil
import concurrent.futures
import sys
import threading
import queue
import json
import re

from pdf2image import convert_from_path
from PIL import Image

try:
    from rich.live import Live
    from rich.table import Table
    from rich.console import Console
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

# --- Global Defaults & Prompts ---
LLAMA_API_URL = "http://127.0.0.1:8033/v1/chat/completions"
BASE_CACHE_DIR = ".medical_cache"
GLOBAL_VLM_CACHE_DIR = os.path.join(BASE_CACHE_DIR, "global_vision_cache")

# The prompt is designed to force structured JSON from the vision model
VISION_PROMPT = """You are a highly accurate medical data extraction AI. Extract the details from this Indian medical prescription/document.
You must reply ONLY with a valid JSON object. Do not include markdown formatting or conversational text.
Format:
{
  "patient_name": "string or null",
  "date": "YYYY-MM-DD or null",
  "doctor_name": "string or null",
  "hospital_clinic": "string or null",
  "diagnoses": ["list of strings"],
  "medications": [
    {"name": "string", "dosage": "string", "frequency": "string"}
  ],
  "raw_extracted_text": "A highly accurate, verbatim text extraction of the entire page."
}
If a field is illegible, use "[UNREADABLE]". If missing, use null."""

# --- Setup Logging ---
os.makedirs("logs", exist_ok=True)
os.makedirs(GLOBAL_VLM_CACHE_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_combined_hash(filepaths, chunk_size=8 * 1024 * 1024):
    """Generates a single SHA-256 hash representing the combined content of all input PDFs."""
    hasher = hashlib.sha256()
    for filepath in sorted(filepaths):
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b''):
                    hasher.update(chunk)
        except FileNotFoundError:
            logger.error(f"File not found for hashing: {filepath}")
            return None
    return hasher.hexdigest()


def parse_url_args(url_list, default_slots=1):
    parsed = []
    for item in url_list:
        parts = item.rsplit(':', 1)
        if len(parts) == 2 and parts[1].isdigit():
            parsed.append((parts[0], int(parts[1])))
        else:
            parsed.append((item, default_slots))
    return parsed


def setup_cache(doc_hash, model_name, synthesis_model_name):
    cache_dir = os.path.join(BASE_CACHE_DIR, doc_hash)
    pages_dir = os.path.join(cache_dir, "pages_cas")
    os.makedirs(pages_dir, exist_ok=True)
    prompt_hash = hashlib.md5(VISION_PROMPT.encode('utf-8')).hexdigest()[:6]
    fingerprint = f"{model_name}_{synthesis_model_name}_p{prompt_hash}"

    return {
        "dir": cache_dir,
        "pages_dir": pages_dir,
        "page_manifest": os.path.join(cache_dir, f"page_manifest_{fingerprint}.json"),
        "extracted_data": os.path.join(cache_dir, f"extracted_data_{fingerprint}.json"),
        "final_timeline": os.path.join(cache_dir, f"final_timeline_{fingerprint}.json")
    }


def robust_api_call(url, payload, max_retries=3, timeout=180):
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(
                url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            duration = time.time() - start_time
            return content, duration
        except requests.exceptions.RequestException as e:
            error_details = f" | Server Details: {e.response.text}" if hasattr(
                e, 'response') and e.response else ""
            if attempt == max_retries - 1:
                logger.error(f"API Failed on {url}: {e}{error_details}")
                raise Exception(f"{e}{error_details}")
            logger.warning(
                f"Network error on attempt {attempt+1}/{max_retries}. Retrying... ({e})")
            time.sleep(2 ** attempt)


def clean_json_response(raw_text):
    """Strips markdown backticks and attempts to parse JSON from the LLM."""
    try:
        # First attempt: parse directly
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Second attempt: Extract via Regex if wrapped in markdown code blocks
        b_ticks = chr(96) * 3
        pattern = b_ticks + r'(?:json)?\s*(\{.*?\})\s*' + b_ticks
        json_match = re.search(pattern, raw_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    # Fallback if the model completely fails to format JSON
    return {
        "patient_name": None, "date": None, "doctor_name": None,
        "hospital_clinic": None, "diagnoses": [], "medications": [],
        "raw_extracted_text": raw_text, "json_parse_error": True
    }


def extract_pdfs_to_cas(pdf_paths, pages_dir, manifest_path):
    """Bursts multiple PDFs to 300 DPI PNGs, hashes them, and creates a combined immutable manifest."""
    if os.path.exists(manifest_path):
        logger.info("PDF Manifest cache hit. Loading existing pages...")
        with open(manifest_path, "r") as f:
            return json.load(f)

    manifest = []
    global_page_counter = 1

    for pdf_path in pdf_paths:
        logger.info(
            f"Extracting PDF pages from {pdf_path} at 300 DPI (This may take a moment)...")
        images = convert_from_path(pdf_path, dpi=300)

        for i, image in enumerate(images, 1):
            temp_path = os.path.join(
                pages_dir, f"temp_{global_page_counter}.png")
            image.save(temp_path, "PNG")

            with open(temp_path, 'rb') as f:
                img_bytes = f.read()

            sha256_hash = hashlib.sha256(img_bytes).hexdigest()
            cas_filename = f"{sha256_hash}.png"
            cas_filepath = os.path.join(pages_dir, cas_filename)

            if not os.path.exists(cas_filepath):
                shutil.move(temp_path, cas_filepath)
            else:
                os.remove(temp_path)

            manifest.append({
                "global_page_num": global_page_counter,
                "source_pdf": os.path.basename(pdf_path),
                "pdf_page_num": i,
                "sha256": sha256_hash,
                "filename": cas_filename
            })
            global_page_counter += 1

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    logger.info(
        f"Successfully extracted {len(manifest)} total pages across all PDFs to CAS storage.")
    return manifest


def process_medical_images(manifest, paths, cluster_nodes, model_name):
    """Distributes pages across VLM cluster to extract structured JSON."""
    if os.path.exists(paths["extracted_data"]):
        logger.info("Structured JSON cache hit. Skipping VLM extraction.")
        with open(paths["extracted_data"], "r") as f:
            return json.load(f)

    url_pool = queue.Queue()
    total_slots = 0
    active_status = {}

    for url, slots in cluster_nodes:
        for i in range(slots):
            url_pool.put(f"{url}|{i+1}")
            active_status[f"{url} (Slot {i+1})"] = "Idle"
            total_slots += 1

    completed_log = []
    status_lock = threading.Lock()
    extracted_results = []

    def generate_table():
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("📄 Processed Pages", style="dim green", width=55)
        table.add_column("⏳ Active Extraction", style="yellow", width=55)
        num_rows = total_slots
        with status_lock:
            recent = completed_log[-num_rows:]
            while len(recent) < num_rows:
                recent.insert(0, "")
            active = [f"[{k.split('//')[-1]}] {v}" for k,
                      v in active_status.items()]
            for c, a in zip(recent, active):
                table.add_row(c, a)
        return table

    def process_page(meta):
        page_num = meta["global_page_num"]
        source_pdf = meta["source_pdf"]
        cas_filepath = os.path.join(paths['pages_dir'], meta["filename"])

        with open(cas_filepath, "rb") as image_file:
            img_bytes = image_file.read()
            base64_image = base64.b64encode(img_bytes).decode('utf-8')

            cache_hasher = hashlib.sha256(img_bytes)
            cache_hasher.update(model_name.encode('utf-8'))
            cache_hasher.update(VISION_PROMPT.encode('utf-8'))
            global_cache_path = os.path.join(
                GLOBAL_VLM_CACHE_DIR, f"{cache_hasher.hexdigest()}.json")

        if os.path.exists(global_cache_path):
            with open(global_cache_path, "r") as f:
                data = json.load(f)
            with status_lock:
                completed_log.append(f"Page {page_num:02d} -> CACHE HIT")
            data["global_page_num"] = page_num
            data["source_pdf"] = source_pdf
            return data

        assigned_slot = url_pool.get()
        target_url, slot_num = assigned_slot.split("|")
        slot_key = f"{target_url} (Slot {slot_num})"

        with status_lock:
            active_status[slot_key] = f"Page {page_num:02d}"

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 2048  # High tokens needed for full text extraction
        }

        try:
            raw_text, duration = robust_api_call(target_url, payload)
            clean_data = clean_json_response(raw_text)

            with open(global_cache_path, "w") as f:
                json.dump(clean_data, f)

            with status_lock:
                active_status[slot_key] = "Idle"
                completed_log.append(
                    f"Page {page_num:02d} -> Extracted ({duration:.1f}s)")

            clean_data["global_page_num"] = page_num
            clean_data["source_pdf"] = source_pdf
            return clean_data
        except Exception as e:
            with status_lock:
                active_status[slot_key] = "Idle"
                completed_log.append(f"Page {page_num:02d} -> FAILED")
            return {"global_page_num": page_num, "source_pdf": source_pdf, "error": str(e)}
        finally:
            url_pool.put(assigned_slot)

    print("\n")
    if RICH_AVAILABLE:
        with Live(get_renderable=generate_table, refresh_per_second=5):
            with concurrent.futures.ThreadPoolExecutor(max_workers=total_slots) as executor:
                extracted_results = list(executor.map(process_page, manifest))
    else:
        logger.info("Running standard parallel extraction...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_slots) as executor:
            extracted_results = list(executor.map(process_page, manifest))

    with open(paths["extracted_data"], "w") as f:
        json.dump(extracted_results, f, indent=4)

    return extracted_results


def interpolate_and_sort_dates(raw_data, paths):
    """Attempts to normalize dates, infers missing dates, and sorts chronologically."""
    logger.info(
        "Interpolating missing dates and organizing chronological timeline...")

    last_known_date = "Unknown"
    for page in sorted(raw_data, key=lambda x: x.get("global_page_num", 0)):
        current_date = page.get("date")

        if current_date and current_date not in ["null", "Unknown", "[UNREADABLE]", None]:
            last_known_date = current_date
            page["interpolated_date"] = current_date
        else:
            page["interpolated_date"] = f"Estimated around {last_known_date}"

    # Sort the array based on interpolated dates (Falling back to the global page sequence)
    sorted_timeline = sorted(raw_data, key=lambda x: (
        str(x.get("interpolated_date", "")), x.get("global_page_num", 0)))

    with open(paths["final_timeline"], "w") as f:
        json.dump(sorted_timeline, f, indent=4)

    logger.info("✅ Combined medical timeline generated successfully.")
    return sorted_timeline


def interactive_qa_loop(timeline_data, synthesis_url, synthesis_model):
    """Boots a REPL loop to chat with the heavy LLM about the patient timeline."""

    print("\n" + "="*70)
    print("🏥 MEDICAL TIMELINE SYNTHESIS COMPLETE")
    print("="*70)
    print(
        f"Connecting to Coordinator Model ({synthesis_model}) on {synthesis_url}")
    print("You may now ask questions about the patient's medical history.")
    print("Type 'exit' or 'quit' to end the session.\n")

    # Convert the massive JSON into a readable string for the system prompt
    context_string = json.dumps(timeline_data, indent=2)

    system_prompt = f"""You are an expert Chief Medical Officer analyzing a patient's historical medical records.
The records have been extracted via OCR and sorted chronologically below.
If a user asks a question, answer it accurately based ONLY on the provided context.
Identify potential drug interactions if asked. 

PATIENT MEDICAL TIMELINE (JSON):
{context_string}"""

    chat_history = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_question = input("\n🩺 Ask a medical question: ")
            if user_question.lower() in ['exit', 'quit']:
                print("Closing medical session. Goodbye!")
                break
            if not user_question.strip():
                continue

            chat_history.append({"role": "user", "content": user_question})

            payload = {
                "model": synthesis_model,
                "messages": chat_history,
                "temperature": 0.1,
                "max_tokens": 4096
            }

            print("Thinking...")
            response_text, duration = robust_api_call(
                synthesis_url, payload, timeout=1800)

            chat_history.append(
                {"role": "assistant", "content": response_text})

            # Make <think> tags visible so the rich Markdown parser doesn't hide them!
            display_text = response_text.replace(
                "<think>", "\n> **🧠 System Thinking:**\n> ").replace("</think>", "\n\n---\n")

            if RICH_AVAILABLE:
                console.print(Markdown(display_text))
            else:
                print(f"\n{display_text}")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error communicating with model: {e}]")
            # Remove the failed message from history so the user can try again
            if len(chat_history) > 1:
                chat_history.pop()


def main():
    parser = argparse.ArgumentParser(
        description="Medical Document AI Pipeline")
    parser.add_argument("--pdfs", type=str, nargs='+', required=True,
                        help="Path(s) to one or more PDF medical records (e.g. file1.pdf file2.pdf or *.pdf).")

    parser.add_argument("--main-urls", type=str, nargs='+', default=[LLAMA_API_URL],
                        help="Worker LLM servers used for parallel Vision processing. Optional slot count: e.g. URL:4")

    parser.add_argument("--model", type=str, default="qwen3-vl",
                        help="The Vision Model expected by the API (default: qwen3-vl)")

    parser.add_argument("--synthesis-model", type=str, required=True,
                        help="The heavy Text Model to use for final Q&A (e.g. qwen3.5-35b)")
    parser.add_argument("--synthesis-url", type=str, required=True,
                        help="Dedicated LLM server endpoint for the text synthesis (e.g. port 8034).")

    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete existing cache for this file batch.")

    args = parser.parse_args()

    # --- Initial Setup ---
    main_parsed = parse_url_args(args.main_urls, default_slots=1)

    # Create a single combined hash representing ALL inputted PDFs
    bundle_hash = get_combined_hash(args.pdfs)
    if not bundle_hash:
        logger.error("❌ One or more files not found.")
        sys.exit(1)

    logger.info("=== Starting Medical Document AI Pipeline ===")
    logger.info(f"Target Files: {len(args.pdfs)} PDF(s) provided.")
    logger.info(f"Patient Bundle Hash: {bundle_hash[:12]}")
    logger.info(
        f"Vision Workers: {sum(slots for url, slots in main_parsed)} slots mapped to {args.model}")
    logger.info(
        f"Coordinator: {args.synthesis_url} mapped to {args.synthesis_model}")

    if args.clear_cache:
        cache_dir_to_clear = os.path.join(BASE_CACHE_DIR, bundle_hash)
        if os.path.exists(cache_dir_to_clear):
            logger.info("Wiping existing document bundle cache...")
            shutil.rmtree(cache_dir_to_clear)

    paths = setup_cache(bundle_hash, args.model, args.synthesis_model)

    # --- Phase 1: Multi-PDF to CAS Extraction ---
    manifest = extract_pdfs_to_cas(
        args.pdfs, paths["pages_dir"], paths["page_manifest"])

    # --- Phase 2: VLM JSON Extraction ---
    raw_extracted_data = process_medical_images(
        manifest, paths, main_parsed, args.model)

    # --- Phase 3: Temporal Interpolation ---
    timeline_data = interpolate_and_sort_dates(raw_extracted_data, paths)

    # --- Phase 4: Interactive Analysis ---
    interactive_qa_loop(timeline_data, args.synthesis_url,
                        args.synthesis_model)


if __name__ == "__main__":
    main()
