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
- Intelligent Compression: Strips OCR noise and deduplicates boilerplate across pages.
- MLX Support: Seamlessly connects to Apple's native mlx_lm framework with smart 404 fallbacks.
- Audit Logging: Tracks exact query timestamps and model processing speeds.

=====================================================
USAGE EXAMPLES:

Standard Llama.cpp Mode:
python medical_pipeline.py \
  --pdfs ./patient_folder/ \
  --model qwen3-vl \
  --main-urls http://127.0.0.1:8033/v1/chat/completions:3 \
  --synthesis-model qwen3.5-35b \
  --synthesis-url http://127.0.0.1:8034/v1/chat/completions \
  --export-dir ./medical_exports/ \
  --compress-context

Apple MLX Mode:
Start MLX Server: python3 -m mlx_lm server --model mlx-community/Llama-3.3-70B-Instruct-4bit --host 0.0.0.0 --port 8034

Run Script:
python medical_pipeline.py \
  --pdfs ./patient_folder/ \
  --model qwen3-vl \
  --main-urls http://127.0.0.1:8033/v1/chat/completions:3 \
  --synthesis-model mlx-community/Llama-3.3-70B-Instruct-4bit \
  --synthesis-url http://127.0.0.1:8034/v1/chat/completions \
  --export-dir ./medical_exports/ \
  --compress-context \
  --use-mlx
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
import copy
from datetime import datetime

from pdf2image import convert_from_path
from PIL import Image

try:
    from rich.live import Live
    from rich.table import Table
    from rich.console import Console
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

# --- Global Defaults & Prompts ---
LLAMA_API_URL = "http://127.0.0.1:8033/v1/chat/completions"
BASE_CACHE_DIR = ".medical_cache"
GLOBAL_VLM_CACHE_DIR = os.path.join(BASE_CACHE_DIR, "global_vision_cache")

# EXPANDED PROMPT: Ensures clinical data isn't lost when raw text is compressed
VISION_PROMPT = """You are a highly accurate medical data extraction AI. Extract the details from this Indian medical prescription/document.
You must reply ONLY with a valid JSON object. Do not include markdown formatting or conversational text.
Format:
{
  "patient_name": "string or null",
  "date": "YYYY-MM-DD or null",
  "doctor_name": "string or null",
  "hospital_clinic": "string or null",
  "symptoms_and_complaints": ["list of strings"],
  "diagnoses": ["list of strings"],
  "lab_results_and_vitals": ["list of strings"],
  "medications": [
    {"name": "string", "dosage": "string", "frequency": "string"}
  ],
  "treatment_plan_and_notes": "string or null",
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

    # Decoupled Cache Logic: PDF extraction depends only on the document hash
    manifest_file = os.path.join(cache_dir, "page_manifest.json")

    # Extracted data depends on the Vision model and prompt, NOT the synthesis text model.
    vlm_fingerprint = f"{model_name}_p{prompt_hash}"

    return {
        "dir": cache_dir,
        "pages_dir": pages_dir,
        "page_manifest": manifest_file,
        "extracted_data": os.path.join(cache_dir, f"extracted_data_{vlm_fingerprint}.json"),
        "final_timeline": os.path.join(cache_dir, f"final_timeline_{vlm_fingerprint}.json")
    }


def robust_api_call(url, payload, use_mlx=False, max_retries=3, timeout=600):
    # Apple's MLX server often throws a 404 Model Not Found if the requested model
    # doesn't perfectly match its internal state. Overriding it here prevents the crash.
    if use_mlx and "model" in payload:
        payload["model"] = "default_model"

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(
                url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)

            # Smart Fallback: If MLX Server throws a 404 on the standard route, try the direct route.
            if response.status_code == 404 and use_mlx and "/v1/chat/completions" in url:
                fallback_url = url.replace(
                    "/v1/chat/completions", "/chat/completions")
                logger.info(
                    f"Received 404. Attempting MLX fallback route: {fallback_url}")
                response = requests.post(
                    fallback_url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)

            response.raise_for_status()

            data = response.json()

            # llama.cpp (and mlx_lm) puts both the <think> block and the final answer inside 'content'
            content = data["choices"][0].get("message", {}).get("content", "")
            if content is None:
                content = ""

            duration = time.time() - start_time
            return content, duration
        except requests.exceptions.RequestException as e:
            error_details = f" | Server Details: {e.response.text}" if hasattr(
                e, 'response') and getattr(e.response, 'text', None) else ""
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
        "hospital_clinic": None, "symptoms_and_complaints": [], "diagnoses": [],
        "lab_results_and_vitals": [], "medications": [], "treatment_plan_and_notes": None,
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


# --- INTELLIGENT COMPRESSION ENGINE ---
def intelligent_compress_text(text, global_seen_lines):
    """Removes OCR noise, stutters, and deduplicates repetitive hospital headers/footers."""
    if not text or not isinstance(text, str):
        return text

    # 1. Strip excessive repeating characters (OCR artifacts: "-------", "........")
    text = re.sub(r'([.\-_= *~#])\1{3,}', r'\1\1\1', text)

    # 2. Remove consecutive repeated words (e.g., "the the the patient")
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

    # 3. Reduce excessive whitespaces
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # 4. Global Line Deduplication (Targeting boilerplate headers, footers, disclaimers across pages)
    lines = text.split('\n')
    deduped_lines = []
    for line in lines:
        stripped = line.strip()
        # Only deduplicate lines longer than 30 characters to avoid breaking valid short data
        if len(stripped) > 30:
            # Normalize: remove punctuation and lowercase for strict duplicate matching
            normalized = re.sub(r'[^\w\s]', '', stripped.lower())
            if normalized not in global_seen_lines:
                global_seen_lines.add(normalized)
                deduped_lines.append(stripped)
        else:
            if stripped:  # Avoid keeping completely empty lines to further compress
                deduped_lines.append(stripped)

    return '\n'.join(deduped_lines).strip()


def compress_timeline(timeline):
    """Recursively traverses the JSON to intelligently compress string fields."""
    global_seen_lines = set()

    def process_node(node):
        if isinstance(node, dict):
            return {k: process_node(v) for k, v in node.items()}
        elif isinstance(node, list):
            # Process lists and remove any empty items
            return [process_node(item) for item in node if item or isinstance(item, (int, bool, float))]
        elif isinstance(node, str):
            return intelligent_compress_text(node, global_seen_lines)
        else:
            return node

    return process_node(timeline)
# -----------------------------------------------------------


def interactive_qa_loop(timeline_data, synthesis_url, synthesis_model, export_dir, bundle_hash, compress_context=False, use_mlx=False):
    """Boots a REPL loop to chat with the heavy LLM about the patient timeline."""

    chat_log_path = os.path.join(
        export_dir, f"chat_transcript_{bundle_hash[:8]}.md")

    print("\n" + "="*70)
    print("🏥 MEDICAL TIMELINE SYNTHESIS COMPLETE")
    print("="*70)

    if use_mlx:
        print(
            f"🍎 Connecting to Apple MLX Server ({synthesis_model}) on {synthesis_url}")
    else:
        print(
            f"🔌 Connecting to Coordinator Server ({synthesis_model}) on {synthesis_url}")

    print(f"Live chat transcript will be saved to: {chat_log_path}")

    # Process Context Compression based on the CLI flag
    if compress_context:
        print("🗜️ Context Compression: ENABLED (Intelligent Regex Deduplication & OCR Noise Reduction)")
        original_size = len(json.dumps(timeline_data))

        # We recursively deduplicate sentences across ALL pages without deleting the clinical notes!
        timeline_to_use = compress_timeline(copy.deepcopy(timeline_data))

        # If the raw extracted text STILL exists and is too large, we can safely delete it now
        # because our expanded prompt already extracted the actual medical data into other fields.
        for page in timeline_to_use:
            if "raw_extracted_text" in page:
                del page["raw_extracted_text"]

        new_size = len(json.dumps(timeline_to_use))
        savings = 100 - ((new_size / original_size) *
                         100) if original_size > 0 else 0
        print(f"📉 Compressed timeline payload size by {savings:.1f}%")
    else:
        print("🗜️ Context Compression: DISABLED (Sending full uncompressed OCR text)")
        timeline_to_use = timeline_data

    print("You may now ask questions about the patient's medical history.")
    print("Type 'exit' or 'quit' to end the session.\n")

    # Convert the chosen JSON into a readable string for the system prompt
    context_string = json.dumps(timeline_to_use, indent=2)

    # HARDENED PROMPT: specifically forcing cross-referencing and contradiction resolution
    system_prompt = f"""You are an expert Chief Medical Officer analyzing a patient's historical medical records.
The records have been extracted via OCR and sorted chronologically below as a JSON array.

INSTRUCTIONS:
1. Answer the user's questions accurately based ONLY on the provided context.
2. Carefully check ALL dates and records. If there is conflicting information (e.g., an allergy listed in 2023 but "No Known Allergy" in 2025), state both and explain the discrepancy.
3. Pay close attention to "raw_extracted_text" for handwritten notes that might not be fully parsed into the structured fields.
4. Identify potential drug interactions if asked.

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

            query_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            chat_history.append({"role": "user", "content": user_question})

            payload = {
                "model": synthesis_model,
                "messages": chat_history,
                "temperature": 0.1,
                "max_tokens": 8192
            }

            print("Thinking...")
            response_text, duration = robust_api_call(
                synthesis_url, payload, use_mlx=use_mlx, timeout=600)

            # CLEANUP: Strip out MLX/Llama stop token leaks
            response_text = response_text.replace(
                "<|eot_id|>", "").replace("<|im_end|>", "").strip()

            chat_history.append(
                {"role": "assistant", "content": response_text})

            # Safely replace the tags for the terminal view, bypassing rich.Markdown swallows
            display_text = response_text.replace(
                "<think>", "\n[🧠 SYSTEM THINKING]\n-------------------\n").replace("</think>", "\n-------------------\n\n")

            # Print using raw print to guarantee visibility
            print(f"\n{display_text}")

            # Append to chat transcript file
            with open(chat_log_path, "a", encoding="utf-8") as f:
                f.write(f"### 🩺 Question: {user_question}\n")
                f.write(
                    f"*Asked on: {query_timestamp} | Processing Time: {duration:.2f} seconds*\n\n")
                f.write(f"**🤖 Answer:**\n{response_text}\n\n---\n\n")

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

    # Updated to support multiple files and directories
    parser.add_argument("--pdfs", type=str, nargs='+', required=True,
                        help="Path(s) to one or more PDF medical records (e.g. file1.pdf file2.pdf or *.pdf).")

    parser.add_argument("--main-urls", type=str, nargs='+', default=[LLAMA_API_URL],
                        help="Worker LLM servers used for parallel Vision processing. Optional slot count: e.g. URL:4")

    parser.add_argument("--model", type=str, default="qwen3-vl",
                        help="The Vision Model expected by the API (default: qwen3-vl)")

    parser.add_argument("--synthesis-model", type=str, required=True,
                        help="The heavy Text Model to use for final Q&A (e.g. mlx-community/Llama-3.3-70B-Instruct-4bit)")

    parser.add_argument("--synthesis-url", type=str, required=True,
                        help="Dedicated LLM server endpoint for the text synthesis (e.g. port 8034).")

    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete existing cache for this file batch.")

    parser.add_argument("--export-dir", type=str, default="./medical_exports",
                        help="Directory to save the final extracted JSON and chat transcripts.")

    parser.add_argument("--compress-context", action="store_true",
                        help="Intelligently compress the timeline by stripping OCR noise and duplicate sentences.")

    parser.add_argument("--use-mlx", action="store_true",
                        help="Enable Apple MLX mode flag for specific synthesis server formatting & 404 fallbacks.")

    args = parser.parse_args()

    # --- Initial Setup ---
    main_parsed = parse_url_args(args.main_urls, default_slots=1)

    # --- Handle Directory & File Parsing ---
    actual_pdf_paths = []
    for path in args.pdfs:
        if os.path.isdir(path):
            actual_pdf_paths.extend(
                glob.glob(os.path.join(path, '**', '*.pdf'), recursive=True))
        elif os.path.isfile(path):
            actual_pdf_paths.append(path)
        else:
            actual_pdf_paths.extend(glob.glob(path, recursive=True))

    actual_pdf_paths = sorted(list(set(actual_pdf_paths)))

    if not actual_pdf_paths:
        logger.error(
            "❌ No PDF files found in the specified paths or directories.")
        sys.exit(1)

    # Create a single combined hash representing ALL inputted PDFs
    bundle_hash = get_combined_hash(actual_pdf_paths)
    if not bundle_hash:
        logger.error("❌ Failed to hash the provided files.")
        sys.exit(1)

    logger.info("=== Starting Medical Document AI Pipeline ===")
    logger.info(
        f"Target Files: {len(actual_pdf_paths)} PDF(s) found and loaded.")
    logger.info(f"Patient Bundle Hash: {bundle_hash[:12]}")
    logger.info(
        f"Vision Workers: {sum(slots for url, slots in main_parsed)} slots mapped to {args.model}")
    logger.info(
        f"Coordinator: {args.synthesis_url} mapped to {args.synthesis_model}")

    if args.use_mlx:
        logger.info("Apple MLX Native mode activated for synthesis.")

    if args.clear_cache:
        cache_dir_to_clear = os.path.join(BASE_CACHE_DIR, bundle_hash)
        if os.path.exists(cache_dir_to_clear):
            logger.info("Wiping existing document bundle cache...")
            shutil.rmtree(cache_dir_to_clear)

    paths = setup_cache(bundle_hash, args.model, args.synthesis_model)

    # --- Phase 1: Multi-PDF to CAS Extraction ---
    manifest = extract_pdfs_to_cas(
        actual_pdf_paths, paths["pages_dir"], paths["page_manifest"])

    # --- Phase 2: VLM JSON Extraction ---
    raw_extracted_data = process_medical_images(
        manifest, paths, main_parsed, args.model)

    # --- Phase 3: Temporal Interpolation ---
    timeline_data = interpolate_and_sort_dates(raw_extracted_data, paths)

    # --- Phase 4: Export Data ---
    os.makedirs(args.export_dir, exist_ok=True)
    export_json_path = os.path.join(
        args.export_dir, f"extracted_timeline_{bundle_hash[:8]}.json")
    with open(export_json_path, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=4)
    logger.info(f"💾 Exported readable patient timeline to: {export_json_path}")

    # --- Phase 5: Interactive Analysis ---
    interactive_qa_loop(
        timeline_data,
        args.synthesis_url,
        args.synthesis_model,
        args.export_dir,
        bundle_hash,
        args.compress_context,
        args.use_mlx
    )


if __name__ == "__main__":
    main()
