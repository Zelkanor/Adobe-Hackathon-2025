import os
import json
import csv
import re
from tqdm import tqdm
from common_features import extract_features

ANNOTATIONS_DIR = r'c:\Users\Sanoja\Desktop\Adobe\Adobe-Hackathon-2025\Datasets\DocBank\docbank_training_data_gpu\annotations'
OUTPUT_CSV = 'docbank_group2.csv'
VERTICAL_MERGE_THRESHOLD = 20

def filter_obj(o):
    text = o.get('text', '').strip()
    if not text:
        return False
    # Removes all variants like ##LTLine##, ###LTIMAGE### etc.
    if re.match(r'^#+LT', text):
        return False
    return True

def merge_blocks(group):
    if not group:
        return None

    # Only keep objects passing the filter
    texts = [obj['text'].strip() for obj in group if isinstance(obj['text'], str) and filter_obj(obj)]
    if not texts:
        return None

    merged_text = ' '.join(texts)
    # Still skip if merged_text is just a sequence of LT tokens
    if not merged_text or re.match(r'^(#+LT\w+#+\s*)+$', merged_text):
        return None

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

    # Filter out unwanted rows, including blank and LT noise tokens
    objs = [o for o in objs if isinstance(o.get('text'), str) and filter_obj(o)]
    objs.sort(key=lambda o: (o.get('label', ''), (o.get('bbox') or o.get('box') or [0, 0, 0, 0])[1]))

    page_font_sizes = []
    merged_blocks = []
    temp_group = []
    last_y = last_label = None
    for obj in objs:
        bbox = obj.get('bbox') or obj.get('box') or [0, 0, 0, 0]
        label = obj.get('label', '')
        y_top = bbox[1]

        if not temp_group or label != last_label or abs(y_top - last_y) > VERTICAL_MERGE_THRESHOLD:
            if temp_group:
                merged = merge_blocks(temp_group)
                if merged:
                    merged_blocks.append(merged)
                    # For font size collection
                    bbox_height = merged['bbox'][3] - merged['bbox'][1]
                    word_count = len([w for w in merged['text'].split() if w])
                    estimated_font_size = bbox_height / word_count if word_count > 0 else bbox_height
                    estimated_font_size = min(max(estimated_font_size, 5), 50)
                    page_font_sizes.append(estimated_font_size)
            temp_group = []
        temp_group.append(obj)
        last_y = y_top
        last_label = label

    if temp_group:
        merged = merge_blocks(temp_group)
        if merged:
            merged_blocks.append(merged)
            bbox_height = merged['bbox'][3] - merged['bbox'][1]
            word_count = len([w for w in merged['text'].split() if w])
            estimated_font_size = bbox_height / word_count if word_count > 0 else bbox_height
            estimated_font_size = min(max(estimated_font_size, 5), 50)
            page_font_sizes.append(estimated_font_size)

    # Feature extraction with prev/next whitespace logic
    rows = []
    for i, merged in enumerate(merged_blocks):
        prev_merged = merged_blocks[i-1] if i > 0 else None
        next_merged = merged_blocks[i+1] if i < len(merged_blocks)-1 else None

        features = extract_features(
            text=merged['text'],
            bold=merged['bold'],
            italic=merged['italic'],
            underline=merged['underline'],
            bbox=merged['bbox'],
            prev_obj=prev_merged,
            next_obj=next_merged,  # <-- pass here!
            page_width=merged['page_width'],
            page_height=merged['page_height'],
            lang='en',
            page_font_sizes=page_font_sizes,
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
