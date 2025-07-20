import re
import numpy as np

def extract_features(
    text, bold, italic, underline, bbox, prev_obj,
    page_width, page_height, lang, all_font_sizes, page_number, label
):
    # Extract bounding box
    if not isinstance(bbox, list) or len(bbox) != 4:
        x1 = y1 = x2 = y2 = 0
    else:
        x1, y1, x2, y2 = bbox

    bbox_width = max(x2 - x1, 1)
    bbox_height = max(y2 - y1, 1)
    area = max(page_width * page_height, 1)

    # Word features
    words = text.split()
    word_count = len(words)

    # --- Estimated Font Size ---
    estimated_font_size = bbox_height / word_count if word_count > 0 else bbox_height
    estimated_font_size = min(max(estimated_font_size, 5), 50)  # Clamp to realistic range

    all_font_sizes.append(estimated_font_size)
    relative_font_size = estimated_font_size / (np.mean(all_font_sizes) + 1e-6)

    try:
        font_size_rank = sorted(set(all_font_sizes), reverse=True).index(estimated_font_size) + 1
    except ValueError:
        font_size_rank = 0

    # Capitalization
    letters = re.sub(r'[^a-zA-Z]', '', text)
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
    title_case_ratio = sum(1 for w in words if w.istitle()) / word_count if word_count else 0
    has_numbering = bool(re.match(r'^[\(\[]?\d+[\.\):\]]?', text.strip()))
    punctuation = int(bool(re.search(r'[.,:;!?]', text)))

    # Whitespace
    whitespace_above = y1 - prev_obj['bbox'][3] if prev_obj and 'bbox' in prev_obj else 0
    whitespace_below = 0

    # Alignment
    center_x = (x1 + x2) / 2
    if center_x < page_width * 0.33:
        alignment = 'left'
    elif center_x > page_width * 0.66:
        alignment = 'right'
    else:
        alignment = 'center'

    # Relative position
    relative_position = (y1 + y2) / 2 / page_height
    position_in_page = f"{int(relative_position * 100)}%"

    # Heading levels
    heading_level = ""
    if font_size_rank == 1:
        heading_level = "H1"
    elif font_size_rank == 2:
        heading_level = "H2"
    elif font_size_rank == 3:
        heading_level = "H3"

    # Combine with document label
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
        "whitespace_below": whitespace_below,
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
