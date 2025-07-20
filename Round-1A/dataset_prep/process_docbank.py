import os
import json
import csv
from tqdm import tqdm
from common_features import extract_features

ANNOTATIONS_DIR = r'c:\Users\Sanoja\Desktop\Adobe\Adobe-Hackathon-2025\Datasets\DocBank\docbank_training_data_gpu\annotations'
OUTPUT_CSV = 'docbank_features.csv'

def extract_all_font_sizes(objs):
    font_sizes = []
    for obj in objs:
        size = obj.get('font') or obj.get('size')
        try:
            size = float(size)
            font_sizes.append(size)
        except:
            continue
    return font_sizes

def process_json_file(file_path, file_id):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Failed to decode JSON in file: {file_path}")
            return []

    objs = []
    page_dims = []

    # Handle dict format
    if isinstance(json_data, dict) and 'pages' in json_data:
        for page in json_data['pages']:
            page_objs = page.get('objs', [])
            objs.extend(page_objs)
            page_dims.append((page.get('width', 1024), page.get('height', 1024)))
    elif isinstance(json_data, list):
        objs = json_data
        page_dims = [(1024, 1024)]  # fallback
    else:
        print(f"⚠️ Unknown format in file: {file_path}")
        return []

    all_font_sizes = extract_all_font_sizes(objs)
    rows = []
    prev_obj = None

    for i, obj in enumerate(objs):
        text = obj.get('text', '')
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue

        # Try bbox from multiple keys
        bbox = obj.get('bbox') or obj.get('box') or [0, 0, 0, 0]
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0, 0, 0, 0]

        font_size = obj.get('font') or obj.get('size', 12)
        bold = obj.get('bold', False)
        italic = obj.get('italic', False)
        underline = obj.get('underline', False)
        label = obj.get('label', '')

        page_width, page_height = page_dims[0] if page_dims else (1024, 1024)

        features = extract_features(
            text=text,
            font_size=font_size,
            bold=bold,
            italic=italic,
            underline=underline,
            bbox=bbox,
            prev_obj=prev_obj,
            page_width=page_width,
            page_height=page_height,
            lang='en',
            all_font_sizes=all_font_sizes,
            page_number=1  # You can update this if your format includes page_id
        )

        features['label'] = label
        features['FileID'] = file_id
        rows.append(features)

        prev_obj = {'bbox': bbox}

    return rows

def process_all_annotations():
    all_rows = []
    json_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.json')]
    print(f"🔍 Looking in: {ANNOTATIONS_DIR}")
    print(f"📄 Processing {len(json_files)} JSON files...")

    for file_name in tqdm(json_files):
        file_path = os.path.join(ANNOTATIONS_DIR, file_name)
        file_id = os.path.splitext(file_name)[0]
        rows = process_json_file(file_path, file_id)
        all_rows.extend(rows)

    if all_rows:
        keys = list(all_rows[0].keys())
        try:
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"\n✅ Done. CSV saved to: {OUTPUT_CSV} ({len(all_rows)} rows)")
        except PermissionError:
            print(f"\n❌ Error: Permission denied. Close Excel if file is open: {OUTPUT_CSV}")
    else:
        print("\n⚠️ No data extracted. CSV not created.")

if __name__ == "__main__":
    process_all_annotations()
