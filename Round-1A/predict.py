import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
import os
import re # Import regex for has_numbering

# --- Configuration & Artifact Loading ---
# This script assumes it's in the project root directory.
# The model artifacts are in a subdirectory named 'final_heading_model'.
MODEL_DIR = Path(__file__).resolve().parent / 'final_heading_model'
logger = print

logger("🔄 Loading model and artifacts...")
try:
    LGBM_MODEL = lgb.Booster(model_file=str(MODEL_DIR / 'lgbm_model.txt'))
    TFIDF_VECTORIZER = joblib.load(MODEL_DIR / 'tfidf_vectorizer.joblib')
    NUM_FEAT_COLS = joblib.load(MODEL_DIR / 'numerical_features.joblib')
    CAT_FEAT_COLS = joblib.load(MODEL_DIR / 'categorical_features.joblib')
    DUMMY_COLUMNS = joblib.load(MODEL_DIR / 'dummy_columns.joblib')
    label_map_encoded = joblib.load(MODEL_DIR / 'label_map.joblib')
    LABEL_MAP_DECODED = {v: k for k, v in label_map_encoded.items()}
    logger("✅ All artifacts loaded successfully.")
except FileNotFoundError as e:
    logger(f"❌ Error: Model artifacts not found in '{MODEL_DIR}'. Please ensure the training was successful and all artifacts exist.")
    logger(f"Details: {e}")
    exit()

def extract_features_from_pdf(pdf_path: Path) -> pd.DataFrame:
    """
    Extracts text blocks and their features from a given PDF file.
    This function replicates the feature engineering from the training data creation.
    """
    blocks = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger(f"   ⚠️ Could not open or process PDF: {pdf_path}. Error: {e}")
        return pd.DataFrame()

    for page_num, page in enumerate(doc):
        page_width, page_height = page.rect.width, page.rect.height
        if page_height == 0: continue

        page_blocks = page.get_text("dict", sort=True)["blocks"]

        for i, block in enumerate(page_blocks):
            if block.get("lines"):
                text = " ".join([line["spans"][0]["text"] for line in block["lines"] if line.get("spans")])
                if not text.strip():
                    continue

                bbox = block["bbox"]
                x1, y1, x2, y2 = bbox
                
                first_span = block["lines"][0]["spans"][0]
                font_size = first_span["size"]
                flags = first_span["flags"]
                words = text.split()
                word_count = len(words)
                
                center_x = (x1 + x2) / 2
                if abs(center_x - page_width / 2) < (page_width * 0.15):
                    alignment = 'center'
                elif x1 < (page_width * 0.1):
                    alignment = 'left'
                else:
                    alignment = 'other'

                whitespace_above = y1 - blocks[-1]["bbox_y2"] if blocks and blocks[-1]["page_number"] == page_num + 1 else y1
                
                blocks.append({
                    'text': text, 'font_size': font_size, 'bold': bool(flags & 2**4),
                    'italic': bool(flags & 2**1), 'underline': bool(flags & 2**3),
                    'word_count': word_count,
                    'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1e-6),
                    'title_case_ratio': sum(1 for w in words if w.istitle()) / (word_count + 1e-6),
                    'has_numbering': bool(re.match(r'^\s*(\d+[\.\)]|\w[\.\)]|\*|\-)', text)),
                    'punctuation': sum(1 for c in text if c in '.,;:!?()[]{}'),
                    'whitespace_above': whitespace_above, 'bbox_x1': x1, 'bbox_y1': y1,
                    'bbox_x2': x2, 'bbox_y2': y2, 'bbox_width': x2 - x1, 'bbox_height': y2 - y1,
                    'relative_position': y1 / page_height, 'alignment': alignment,
                    'page_number': page_num + 1, 'font_size_rank': 0,
                    'relative_font_size': 1.0, 'whitespace_below': 0
                })

    if not blocks: return pd.DataFrame()
    df = pd.DataFrame(blocks)
    
    df = df.sort_values(by=['page_number', 'bbox_y1']).reset_index(drop=True)
    df['whitespace_below'] = df['bbox_y1'].shift(-1) - df['bbox_y2']
    df['whitespace_below'] = df['whitespace_below'].fillna(0).apply(lambda x: max(0, x))
    
    for page in df['page_number'].unique():
        page_df = df[df['page_number'] == page]
        if page_df.empty: continue
        median_font = page_df['font_size'].median()
        size_to_rank = {size: rank + 1 for rank, size in enumerate(sorted(page_df['font_size'].unique(), reverse=True))}
        df.loc[page_df.index, 'font_size_rank'] = page_df['font_size'].map(size_to_rank)
        df.loc[page_df.index, 'relative_font_size'] = page_df['font_size'] / median_font if median_font > 0 else 1.0

    return df

def find_document_title(df: pd.DataFrame) -> (str, pd.Index):
    """
    Heuristic: on page 1, pick the top‐of‐page blocks that
    1) use font_size ≥ 90% of max
    2) span ≥ 50% of page width
    3) lie within the top 25% of the page
    4) are not obvious noise (Version, Page, www, ©)
    Then cluster any that are within 1.5× the median line-height
    and join their text in reading order.
    """
    if df.empty:         
        return "", pd.Index([])
    # restrict to page 1, near the top
    p1 = df[df.page_number == 1].copy()
    if p1.empty:
        return "", pd.Index([])
    
    # page_w can be inferred from the max x2 coordinate
    page_w = p1['bbox_x2'].max()
    max_font = p1['font_size'].max()
    
    # basic filters
    cand = p1[
        (p1['font_size'] >= max_font * 0.9) &
        (p1['bbox_width'] >= page_w * 0.5) &
        (p1['relative_position'] < 0.25) &
        (~p1['text'].str.contains(r'www\.|http|©|Page\s+\d+', regex=True))
    ].copy()
    
    if cand.empty:
        return "", pd.Index([])
        
    # compute a median line height for vertical clustering
    median_h = cand['bbox_height'].median()
    
    # sort by y1 (top of page is smaller y1)
    cand = cand.sort_values('bbox_y1').reset_index()
    
    groups = []
    current = [cand.loc[0]]
    for idx in range(1, len(cand)):
        row = cand.loc[idx]
        prev = current[-1]
        # if this block starts “close” to the previous block
        if row['bbox_y1'] - prev['bbox_y2'] < median_h * 1.5:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    groups.append(current)
    
    # score each group by sum(font_size) × average title_case_ratio
    best_group = max(groups, key=lambda grp: sum(r['font_size'] for r in grp) * \
                                     (sum(r['title_case_ratio'] for r in grp) / len(grp)))
                                     
    # collect their original indices and texts
    indices = [int(r['index']) for r in best_group]
    
    # join text in reading order (by bbox_x1)
    best_group_sorted = sorted(best_group, key=lambda r: r['bbox_x1'])
    title_text = " ".join(r['text'].strip() for r in best_group_sorted)
    
    # final sanity: drop insanely short or all-caps “logos”
    words = title_text.split()
    if len(words) <= 2 and sum(1 for c in title_text if c.isupper()) / (len(title_text)+1e-6) > 0.9:
        return "", pd.Index([])
        
    return title_text, pd.Index(indices)

# --- NEW HEURISTIC LOGIC (AS PER YOUR SUGGESTION) ---

def detect_document_type(df: pd.DataFrame) -> str:
    if df.empty:
        return 'standard'
    # ----- detect obvious multipage reports first -----
    if len(df.page_number.unique()) > 2:
        return 'standard'
    # ----- “form” heuristic (short lines + many colons laid out in table form) -----
    short_lines = (df['word_count'] <= 6).mean()
    colon_ends  = df['text'].str.strip().str.endswith(':').mean()
    has_underlines = (df['underline'] == True).mean()
    if short_lines > .45 and (colon_ends > .25 or has_underlines > .10):
        # BUT: forms almost always have > 15 blocks *ending* with a colon.
        # Invitations normally have only a handful, so veto the form label
        # if there are fewer than 15 such “label” lines
        if (df['text'].str.strip().str.endswith(':').sum() >= 15):
            return 'form'
    # ----- “stylized” = single page + wide font dispersion -----
    if len(df.page_number.unique()) == 1:
        font_min = df['font_size'].min()
        if font_min > 0:
            dispersion = df['font_size'].max() / font_min
            if dispersion > 1.6:
                return 'stylized'
    return 'standard'

def _looks_like_body_paragraph(txt: str, wc: int) -> bool:
    """
    True  → definitely NOT a heading
    """
    if wc > 25:
        return True
    # very few headings contain three commas or a full stop in the first 50 chars
    if txt.count(',') >= 3 or txt[:50].count('.') >= 2:
        return True
    # sentences that start with pronouns/verbs are rarely headings
    if re.match(r'^\s*(we|it|the|a|an|to|for)\b', txt, flags=re.I):
        return True
    return False

def _pre_filter_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes anything that *cannot* be a heading.
    Call once inside generate_hackathon_output() just after content_df is built.
    """
    mask = (
        (~df['text'].str.contains(r'www\.|http|\.(com|org|net)', case=False, regex=True)) &
        (~df['text'].str.contains(r'\d{3,5}\s+\w+\s+\w+')) &   # street addresses
        (~df['text'].str.contains(r'\(\d{3}\)\s*\d{3}[-\s]\d{4}')) &   # phone numbers
        (~df.apply(lambda r: _looks_like_body_paragraph(r['text'], r['word_count']), axis=1))
    )
    return df[mask].copy()

LABEL_REGEX  = re.compile(
    r'\b(ADDRESS|PHONE|EMAIL|CONTACT|RSVP|DATE|TIME|LOCATION|WHEN|WHERE|NAME'
    r'|DESIGNATION|AGE|RELATIONSHIP|PAY|HOME TOWN|AMOUNT)\b:?$', re.I)
ADDRESS_REGEX = re.compile(
    r'\d{1,5}\s+\w+.*\b(ST|STREET|RD|ROAD|AVE|AVENUE|BLVD|DR|TN|TX|CA|NY|ON)\b|\b\d{5}(-\d{4})?$', re.I)
NUMBER_RATIO = lambda txt: len(re.findall(r'\d', txt)) / (len(txt)+1e-6)

def _score_stylized(cand: pd.Series, median_ws: float, max_font: float) -> float:
    """Weighted score; bigger = more likely to be the real H1."""
    score  = 0
    if max_font > 0: score += 3.0 * (cand.font_size / max_font)
    score += 1.5 if cand.alignment == 'center' else 0.5
    if median_ws > 0: score += 1.0 if cand.whitespace_above > 1.5*median_ws else 0
    score += 1.0 if '!' in cand.text or re.search(r'\b(HOPE|WELCOME|JOIN|PARTY|CONGRAT)', cand.text, re.I) else 0
    score -= 1.0 if cand.word_count <= 2 else 0
    score -= 1.5 if NUMBER_RATIO(cand.text) > .30 else 0
    return score

def generate_hackathon_output(feature_df: pd.DataFrame) -> str:
    document_title, title_index = find_document_title(feature_df)
    content_df = feature_df.drop(index=title_index)
    content_df = _pre_filter_blocks(content_df)
    doc_type   = detect_document_type(feature_df)
    
    if doc_type == 'form':
        logger("   Detected a form-like document. Applying strict heading rules.")
        return json.dumps({"title": document_title, "outline": []}, indent=4, ensure_ascii=False)
    
    if doc_type == 'stylized':
        logger("   Detected a stylized document. Using enhanced heading rules.")
        
        MIN_FS = content_df['font_size'].quantile(0.60)
        candidates = content_df[content_df['font_size'] >= MIN_FS]
        
        if candidates.empty:
            return json.dumps({"title": document_title, "outline": []}, indent=4, ensure_ascii=False)
            
        max_font = candidates['font_size'].max()
        median_ws = candidates['whitespace_above'].median()
        candidates['score'] = candidates.apply(_score_stylized, axis=1, median_ws=median_ws, max_font=max_font)
        
        best = candidates.loc[[candidates['score'].idxmax()]]
        
        outline = [{"level": "H1",
                    "text": best.iloc[0]['text'],
                    "page": int(best.iloc[0]['page_number']) - 1}]
        return json.dumps({"title": document_title, "outline": outline}, indent=4, ensure_ascii=False)

    # --- STANDARD DOCUMENT: USE ML MODEL ---
    if content_df.empty:
        return json.dumps({"title": document_title, "outline": []}, indent=4, ensure_ascii=False)

    for col in NUM_FEAT_COLS + CAT_FEAT_COLS:
        if col not in content_df.columns:
            content_df[col] = 0

    for col in NUM_FEAT_COLS:
        content_df[col] = pd.to_numeric(content_df[col], errors='coerce')

    bool_features_in_num = [f for f in ['bold', 'italic', 'underline', 'has_numbering'] if f in NUM_FEAT_COLS]
    for col in bool_features_in_num:
        content_df[col] = content_df[col].astype(int)
    
    num_feat = csr_matrix(content_df[NUM_FEAT_COLS].fillna(0).values)
    
    cat_feat_df = pd.get_dummies(content_df[CAT_FEAT_COLS], dtype=float)
    cat_feat_df = cat_feat_df.reindex(columns=DUMMY_COLUMNS, fill_value=0)
    cat_feat = csr_matrix(cat_feat_df)
    
    text_feat = TFIDF_VECTORIZER.transform(content_df['text'].astype(str))
    
    final_features = hstack([num_feat, cat_feat, text_feat], format='csr')

    predictions = np.argmax(LGBM_MODEL.predict(final_features), axis=1)
    
    outline = [{
        "level": LABEL_MAP_DECODED.get(pred, 'Other'),
        "text": row['text'],
        "page": int(row['page_number'])
    } for pred, (_, row) in zip(predictions, content_df.iterrows())]
    
    return json.dumps({"title": document_title, "outline": sorted(outline, key=lambda x: x['page'])}, indent=4, ensure_ascii=False)

def main():
    docker_input_dir = Path("/app/input")
    docker_output_dir = Path("/app/output")
    
    project_root = Path(__file__).resolve().parent
    local_input_dir = project_root / 'input'
    local_output_dir = project_root / 'output'

    if docker_input_dir.exists():
        input_dir = docker_input_dir
        output_dir = docker_output_dir
        logger("Running in Docker mode.")
    else:
        input_dir = local_input_dir
        output_dir = local_output_dir
        logger("Running in local testing mode.")
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

    logger(f"🚀 Starting processing of PDFs in '{input_dir}'...")
    
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger(f"⚠️ No PDF files found in '{input_dir}'. Please add PDFs to process.")
        return

    for pdf_path in pdf_files:
        logger(f"   Processing: {pdf_path.name}")
        features_df = extract_features_from_pdf(pdf_path)
        if features_df.empty:
            logger(f"   ⚠️ No text blocks found in {pdf_path.name}. Skipping.")
            continue
        json_output = generate_hackathon_output(features_df)
        output_path = output_dir / f"{pdf_path.stem}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_output)
        logger(f"   ✅ Saved output to {output_path}")

    logger("🎉 Processing complete.")

if __name__ == '__main__':
    main()
