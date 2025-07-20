import re
import unicodedata


def calc_caps_ratio(text):
    if not text:
        return 0.0
    n_caps = sum(1 for c in text if c.isupper())
    return n_caps / max(1, len(text))


def calc_title_case_ratio(text):
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
    title_case_words = sum(1 for word in words if word.istitle())
    return title_case_words / len(words)


def calc_paragraph_spacing(obj, prev_obj=None):
    if prev_obj:
        return max(0, obj['bbox'][1] - prev_obj['bbox'][3])
    return 0


def detect_script(text):
    for ch in text:
        if ch.strip():
            name = unicodedata.name(ch, '').lower()
            if 'cjk' in name or 'hangul' in name:
                return 'CJK'
            if 'arabic' in name:
                return 'ARABIC'
            if 'hebrew' in name:
                return 'HEBREW'
            if 'devanagari' in name:
                return 'DEVANAGARI'
    return 'LATIN'


def detect_alignment(bbox, page_width):
    margin = 0.1 * page_width
    x_center = (bbox[0] + bbox[2]) / 2
    if abs(x_center - page_width / 2) < margin:
        return 'center'
    elif bbox[0] < margin:
        return 'left'
    elif bbox[2] > page_width - margin:
        return 'right'
    return 'other'


def has_numbering(text):
    return bool(re.match(r'^\s*[\dA-Za-z]+\s{0,1}[\.\)]', text.strip()))


def has_punctuation(text):
    return int(bool(re.search(r'[.?!,:;]', text)))


def extract_features(
    text, font_size, bold, italic, underline, bbox,
    prev_obj=None, page_width=1024, page_height=1024, lang=None,
    all_font_sizes=None, page_number=1
):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    features = {}
    features['text'] = text.strip()
    features['caps_ratio'] = calc_caps_ratio(text)
    features['paragraph_spacing'] = calc_paragraph_spacing({'bbox': bbox}, prev_obj)
    features['font_size'] = font_size
    features['text_length'] = len(text)
    features['bbox_x1'] = x1
    features['bbox_y1'] = y1
    features['bbox_x2'] = x2
    features['bbox_y2'] = y2
    features['bbox_width'] = width
    features['bbox_height'] = height
    features['bold'] = int(bool(bold))
    features['italics'] = int(bool(italic))
    features['underline'] = int(bool(underline))
    features['relative_position'] = y1 / max(1, page_height)
    features['lang'] = lang or detect_script(text)
    features['script'] = detect_script(text)
    features['alignment'] = detect_alignment(bbox, page_width)
    features['has_numbering'] = int(has_numbering(text))
    features['punctuation'] = has_punctuation(text)
    features['page_number'] = page_number
    features['position_in_page'] = y1 / max(1, page_height)
    features['word_count'] = len(text.strip().split())
    features['title_case_ratio'] = calc_title_case_ratio(text)

    # Relative font size and rank
    if all_font_sizes and isinstance(all_font_sizes, list):
        try:
            max_font = max(all_font_sizes)
            unique_sorted = sorted(set(all_font_sizes), reverse=True)
            features['relative_font_size'] = font_size / max(1, max_font)
            features['font_size_rank'] = unique_sorted.index(font_size) + 1
        except Exception:
            features['relative_font_size'] = 1.0
            features['font_size_rank'] = 1
    else:
        features['relative_font_size'] = 1.0
        features['font_size_rank'] = 1

    # Whitespace above/below (for paragraph spacing)
    features['whitespace_above'] = calc_paragraph_spacing({'bbox': bbox}, prev_obj)
    # This could be updated when the next object becomes available
    features['whitespace_below'] = 0

    return features
