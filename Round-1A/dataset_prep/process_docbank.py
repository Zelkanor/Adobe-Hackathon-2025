import os
import json
import csv
from tqdm import tqdm
from common_features import extract_features

ANNOTATIONS_DIR = r'c:\Users\Sanoja\Desktop\Adobe\Adobe-Hackathon-2025\Datasets\DocBank\docbank_training_data_gpu\annotations'
OUTPUT_CSV = 'docbank_group.csv'
VERTICAL_MERGE_THRESHOLD = 20

def merge_blocks(group):
    if not group:
        return None

    texts = [obj['text'].strip() for obj in group if isinstance(obj['text'], str)]
    if not texts:
        return None

    merged_text = ' '.join(texts)
    x1s, y1s, x2s, y2s = zip(*[obj.get('bbox') or obj.get('box') or [0, 0, 0, 0] for obj in group])
    merged_bbox = [min(x1s), min(y1s), max(x2s), max(y2s)]

    bold = any(obj.get('bold', False) for obj in group)
    italic = any(obj.get('italic', False) for obj in group)
    underline = any(obj.get('underline', False) for obj in group)
    label = group[0].get('label', '')
    page_width = group[0].get('page_width', 1024)
    page_height = group[0].get('page_height', 1024)

    return {
        "text": merged_text,
        "bbox": merged_bbox,
        "bold": bold,
        "italic": italic,
        "underline": underline,
        "label": label,
        "page_width": page_width,
        "page_height": page_height
    }

def process_json_file(file_path, file_id):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Failed to decode JSON in file: {file_path}")
            return []

    objs = []
    if isinstance(json_data, dict) and 'pages' in json_data:
        for page in json_data['pages']:
            pw, ph = page.get('width', 1024), page.get('height', 1024)
            for obj in page.get('objs', []):
                obj['page_width'] = pw
                obj['page_height'] = ph
                objs.append(obj)
    elif isinstance(json_data, list):
        objs = json_data
    else:
        return []

    objs = [o for o in objs if isinstance(o.get('text'), str) and o['text'].strip()]
    objs.sort(key=lambda o: (o.get('label', ''), (o.get('bbox') or o.get('box') or [0, 0, 0, 0])[1]))

    all_font_sizes = []
    rows = []
    current_group = []
    last_y = last_label = None

    for obj in objs:
        bbox = obj.get('bbox') or obj.get('box') or [0, 0, 0, 0]
        label = obj.get('label', '')
        y_top = bbox[1]

        if not current_group or label != last_label or abs(y_top - last_y) > VERTICAL_MERGE_THRESHOLD:
            merged = merge_blocks(current_group)
            if merged:
                features = extract_features(
                    text=merged['text'],
                    bold=merged['bold'],
                    italic=merged['italic'],
                    underline=merged['underline'],
                    bbox=merged['bbox'],
                    prev_obj=None,
                    page_width=merged['page_width'],
                    page_height=merged['page_height'],
                    lang='en',
                    all_font_sizes=all_font_sizes,
                    page_number=1,
                    label=merged['label']
                )
                features['FileID'] = file_id
                rows.append(features)
            current_group = []

        current_group.append(obj)
        last_y = y_top
        last_label = label

    if current_group:
        merged = merge_blocks(current_group)
        if merged:
            features = extract_features(
                text=merged['text'],
                bold=merged['bold'],
                italic=merged['italic'],
                underline=merged['underline'],
                bbox=merged['bbox'],
                prev_obj=None,
                page_width=merged['page_width'],
                page_height=merged['page_height'],
                lang='en',
                all_font_sizes=all_font_sizes,
                page_number=1,
                label=merged['label']
            )
            features['FileID'] = file_id
            rows.append(features)

    return rows

def process_all_annotations():
    all_rows = []
    json_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.json')]
    print(f"🔍 Processing {len(json_files)} files...")

    for file_name in tqdm(json_files):
        file_path = os.path.join(ANNOTATIONS_DIR, file_name)
        file_id = os.path.splitext(file_name)[0]
        all_rows.extend(process_json_file(file_path, file_id))

    if all_rows:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n✅ Saved to {OUTPUT_CSV} with {len(all_rows)} rows.")
    else:
        print("\n⚠️ No data extracted.")

if __name__ == "__main__":
    process_all_annotations()
