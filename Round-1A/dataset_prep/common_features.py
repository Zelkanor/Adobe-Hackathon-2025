import re
import numpy as np

def extract_features(
    text, font_size, bold, italic, underline, bbox, prev_obj,
    page_width, page_height, lang, all_font_sizes, page_number
):
    # 1. Safely extract bounding box
    if not isinstance(bbox, list) or len(bbox) != 4:
        x1 = y1 = x2 = y2 = 0
    else:
        x1, y1, x2, y2 = bbox

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    area = page_width * page_height if page_width and page_height else 1

    # 2. Convert font size safely
    try:
        font_size = float(font_size)
    except:
        font_size = 12.0

    # 3. Font size rank and relative comparison
    try:
        font_size_rank = sorted(set(all_font_sizes), reverse=True).index(font_size) + 1
    except:
        font_size_rank = 0

    relative_font_size = font_size / (np.mean(all_font_sizes) + 1e-6) if all_font_sizes else 1.0

    # 4. Capitalization features
    letters = re.sub(r'[^a-zA-Z]', '', text)
    caps = sum(1 for c in letters if c.isupper())
    caps_ratio = caps / len(letters) if letters else 0

    # 5. Word-based features
    words = text.split()
    word_count = len(words)
    title_case_ratio = sum(1 for w in words if w.istitle()) / word_count if word_count else 0
    has_numbering = bool(re.match(r'^\(?\d+[\.\)]', text.strip()))
    punctuation = int(bool(re.search(r'[.,:;!?]', text)))

    # 6. Whitespace above from previous object
    whitespace_above = y1 - prev_obj['bbox'][3] if prev_obj and 'bbox' in prev_obj else 0
    whitespace_below = 0  # Placeholder if needed later

    # 7. Alignment feature (based on horizontal center)
    center_x = (x1 + x2) / 2
    if center_x < page_width * 0.33:
        alignment = 'left'
    elif center_x > page_width * 0.66:
        alignment = 'right'
    else:
        alignment = 'center'

    # 8. Relative vertical position on page
    relative_position = (y1 + y2) / 2 / page_height if page_height else 0.5
    position_in_page = f"{int(relative_position * 100)}%"

    # Return dictionary of all features
    return {
        "text": text,
        "font_size": round(font_size, 2),
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
        "page_number": page_number
    }
