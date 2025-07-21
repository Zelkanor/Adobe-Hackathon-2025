import os
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

PDF_DIR = r"C:\Users\Sanoja\Desktop\Adobe\Adobe-Hackathon-2025\Datasets\DocLayNet\DocLayNetextra\PDF"
OUTPUT_CSV = "doclaynet_group2.csv"
MAX_WORKERS = 4 

def classify_text_block(text, font_size, fontname, italic, page_font_sizes):
    """Classifies a text block as H1, H2, H3, Paragraph, List, Caption."""
    if not page_font_sizes:
        return "Paragraph"

    unique_sizes = sorted(set(page_font_sizes), reverse=True)
    h1_thresh = unique_sizes[0]
    h2_thresh = unique_sizes[1] if len(unique_sizes) > 1 else h1_thresh - 1
    h3_thresh = unique_sizes[2] if len(unique_sizes) > 2 else h2_thresh - 1

    line = text.strip()
    if font_size >= h1_thresh:
        return "H1"
    elif font_size >= h2_thresh:
        return "H2"
    elif font_size >= h3_thresh:
        return "H3"
    elif line.startswith(("-", "•", "*")) or line[:2].isdigit():
        return "List"
    elif len(line) < 50 and italic:
        return "Caption"
    else:
        return "Paragraph"

def process_pdf(file_path_id_tuple):
    import pdfplumber
    from statistics import mean
    from common_features import extract_features

    file_path, file_id = file_path_id_tuple
    rows = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                width, height = page.width, page.height
                words = page.extract_words(extra_attrs=["fontname", "size"])
                # Build lines by y0 (top)
                lines = {}
                for w in words:
                    y0 = round(w['top'], 1)
                    lines.setdefault(y0, []).append(w)
                if not lines:
                    continue
                # === Step 1. Gather per-page font sizes ===
                page_font_sizes = []
                for line_words in lines.values():
                    if not line_words:
                        continue
                    sizes = [w['size'] for w in line_words]
                    # For lines with many words, use average font size per line
                    avg_size = mean(sizes)
                    page_font_sizes.append(avg_size)

                # === Step 2. Prepare merged line features ===
                merged_lines = []
                for y0 in sorted(lines.keys()):
                    line_words = lines[y0]
                    text = " ".join(w['text'] for w in line_words).strip()
                    if not text:
                        continue
                    font_sizes = [w['size'] for w in line_words]
                    avg_size = mean(font_sizes)
                    fontnames = [w.get('fontname', '').lower() for w in line_words]
                    bold = any('bold' in f for f in fontnames)
                    italic = any(('italic' in f or 'oblique' in f) for f in fontnames)
                    underline = False
                    x0 = min(w['x0'] for w in line_words)
                    y_top = min(w['top'] for w in line_words)
                    x1 = max(w['x1'] for w in line_words)
                    y_bottom = max(w['bottom'] for w in line_words)
                    bbox = [x0, y_top, x1, y_bottom]
                    label = classify_text_block(text, avg_size, fontnames[0], italic, page_font_sizes)
                    merged_lines.append({
                        'text': text,
                        'bold': bold,
                        'italic': italic,
                        'underline': underline,
                        'bbox': bbox,
                        'page_width': width,
                        'page_height': height,
                        'lang': 'en',
                        'page_font_sizes': page_font_sizes,  # Pass per-page!
                        'page_number': page_number,
                        'label': label,
                    })

                # === Step 3. Extract features with prev/next block for whitespace ===
                for idx, merged in enumerate(merged_lines):
                    prev_merged = merged_lines[idx-1] if idx > 0 else None
                    next_merged = merged_lines[idx+1] if idx < len(merged_lines)-1 else None
                    feature = extract_features(
                        text=merged['text'],
                        bold=merged['bold'],
                        italic=merged['italic'],
                        underline=merged['underline'],
                        bbox=merged['bbox'],
                        prev_obj=prev_merged,
                        next_obj=next_merged,
                        page_width=merged['page_width'],
                        page_height=merged['page_height'],
                        lang=merged['lang'],
                        page_font_sizes=merged['page_font_sizes'],
                        page_number=merged['page_number'],
                        label=merged['label']
                    )
                    feature["FileID"] = file_id
                    rows.append(feature)
    except Exception as e:
        print(f"⚠️ Failed on {file_id}: {e}")

    return rows

def process_all_pdfs():
    pdf_files = [(os.path.join(PDF_DIR, f), os.path.splitext(f)[0])
                for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    print(f"🔍 Found {len(pdf_files)} PDF files. Starting parallel processing with {MAX_WORKERS} workers...")
    all_rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
            try:
                result = future.result()
                all_rows.extend(result)
            except Exception as e:
                print(f"❌ Error processing {futures[future]}: {e}")
    if all_rows:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n✅ Saved {len(all_rows)} rows to {OUTPUT_CSV}")
    else:
        print("\n⚠️ No features extracted.")

if __name__ == "__main__":
    process_all_pdfs()
