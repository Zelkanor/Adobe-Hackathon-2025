import re
import numpy as np

def extract_features(
    text, bold, italic, underline, bbox, prev_obj, next_obj,
    page_width, page_height, lang, page_font_sizes, page_number, label
):
    if not isinstance(bbox, list) or len(bbox) != 4:
        x1 = y1 = x2 = y2 = 0
    else:
        x1, y1, x2, y2 = bbox

    bbox_width = max(x2 - x1, 1)
    bbox_height = max(y2 - y1, 1)
    area = max(page_width * page_height, 1)

    words = text.split()
    word_count = len(words)

    estimated_font_size = bbox_height / word_count if word_count > 0 else bbox_height
    estimated_font_size = min(max(estimated_font_size, 5), 50)

    unique_font_sizes = sorted(set(page_font_sizes), reverse=True)
    try:
        font_size_rank = unique_font_sizes.index(estimated_font_size) + 1
    except ValueError:
        font_size_rank = 0

    relative_font_size = estimated_font_size / (np.mean(page_font_sizes) + 1e-6)

    letters = re.sub(r'[^a-zA-Z]', '', text)
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
    title_case_ratio = sum(1 for w in words if w.istitle()) / word_count if word_count else 0
    has_numbering = bool(re.match(r'^[\(\[]?\d+[\.\):\]]?', text.strip()))
    punctuation = int(bool(re.search(r'[.,:;!?]', text)))

    whitespace_above = (
        (y1 - prev_obj['bbox'][3]) if prev_obj and 'bbox' in prev_obj and prev_obj['bbox'] else 0
    )
    whitespace_below = (
        (next_obj['bbox'][1] - y2) if next_obj and 'bbox' in next_obj and next_obj['bbox'] else 0
    )

    center_x = (x1 + x2) / 2
    if center_x < page_width * 0.33:
        alignment = 'left'
    elif center_x > page_width * 0.66:
        alignment = 'right'
    else:
        alignment = 'center'

    relative_position = (y1 + y2) / 2 / page_height
    position_in_page = f"{int(relative_position * 100)}%"

    heading_level = ""
    if font_size_rank == 1:
        heading_level = "H1"
    elif font_size_rank == 2:
        heading_level = "H2"
    elif font_size_rank == 3:
        heading_level = "H3"

    combined_label = f"{label}|{heading_level}" if heading_level else label

    return {
        "text": text,
        "font_size": round(estimated_font_size, 2),
        "font_size_rank": font_size_rank,
        "relative_font_size": round(relative_font_size, 3),
        "bold": bool(bold),
        "italic": bool(italic),
        "underline": bool(underline),
        "caps_ratio": round(caps_ratio, 3),
        "word_count": word_count,
        "title_case_ratio": round(title_case_ratio, 3),
        "has_numbering": has_numbering,
        "punctuation": punctuation,
        "whitespace_above": round(whitespace_above, 2),
        "whitespace_below": round(whitespace_below, 2),
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_width": round(bbox_width, 2),
        "bbox_height": round(bbox_height, 2),
        "relative_position": round(relative_position, 3),
        "position_in_page": position_in_page,
        "alignment": alignment,
        "lang": lang,
        "page_number": page_number,
        "heading_level": heading_level,
        "label": combined_label
    }
