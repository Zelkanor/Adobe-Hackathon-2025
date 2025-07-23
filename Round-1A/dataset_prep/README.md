# 📘 Adobe Hackathon 2025 — Round 1A: Dataset Preparation

## 🚀 Goal

The aim of Round 1A is to **understand the structure of documents** by extracting a clean, hierarchical outline — including the **Title**, and **H1, H2, H3** headings — from raw PDFs.

This README outlines how we prepared and extracted features from multiple public document datasets: **DocBank**, **DocLayNet**, and **PubLayNet**, and structured them into a unified CSV for training/analysis.

---

## 📂 Datasets Used

We used three large-scale, diverse document layout datasets:

- **DocBank**: Offers academic and scientific document structures.
- **DocLayNet**: A large annotated dataset of real-world PDF documents containing diverse layouts (text, tables, images).
- **PubLayNet**: Focuses on publications and academic articles with labeled layout elements.

**Why these datasets?**

- They provide structural information aligned with the Round 1A objective: detecting headings and outlines.
- They span multiple layouts and document types, enabling generalization.
- Ideal for supervised heading- level extraction.

---

## 🧱 Dataset Processing Pipeline

- Datasets were **downloaded from official GitHub sources**.
- Each dataset was **processed using a custom script** to extract meaningful blocks and compute structured features.
- The resulting CSV files were then **merged into one file**.

---

## 🛠 Code & Functional Breakdown

### `process_docbank.py`

Processes annotated JSON files from DocBank:

- `filter_obj(obj)`: Removes noisy or placeholder tokens (`##LTLINE##`, empty text).
- `merge_blocks(group)`: Merges vertically aligned, same-label blocks into single line blocks with combined bbox.
- `process_json_file(file_path, file_id)`: Processes each file:
  - Groups and merges blocks
  - Estimates font size
  - Extracts features with contextual (prev/next) spacing
- `process_all_annotations()`: Iterates all annotation JSONs and writes a single CSV file `docbank_group2.csv`.

---

### `process_doclaynet.py`

Processes raw PDF files using `pdfplumber`:

- `classify_text_block(...)`: Uses average font size + italics/numbering/bullets to assign `H1/H2/H3/List/Caption/Paragraph`.
- `process_pdf(file_path, file_id)`: Extracts words, merges them line-wise, estimates font sizes and computes features.
- `process_all_pdfs()`: Batch processes all PDFs using multiprocessing, and outputs `doclaynet_group2.csv`.

---

### `extract_features.py`

Central feature extractor for all datasets. Computes the following:

- **Geometric features**: `bbox`, `width`, `height`, `alignment`, `relative position`
- **Textual features**: `caps_ratio`, `title_case_ratio`, `has_numbering`, `punctuation`
- **Typographic features**: `font_size`, `font_size_rank`, `relative_font_size`, `bold`, `italic`, `underline`
- **Contextual spacing**: `whitespace_above`, `whitespace_below`
- **Label heuristics**: Infers heading level (H1/H2/H3) based on font rank
- Combines all into a dictionary with metadata

---

### `merge_csv.py`

Merged DocBank, DocLayNet CSV outputs into: 

- `append_docbank_to_doclay(...)`: Merges two CSVs (DocLayNet + DocBank) and outputs a unified dataset.

---

## ✅ Final Output

- Merged CSV: `merged_features2.csv`
- Each row = one merged block
- Includes heading level labels and 30+ rich features

---

## 📑 Feature List

| Feature Name           | Description |
|------------------------|-------------|
| `text`                | Raw textual content of the block |
| `font_size`           | Estimated font size from height/word count |
| `font_size_rank`      | Rank of font size on the page |
| `relative_font_size`  | Ratio to average font size |
| `bold`                | Is any text bold? |
| `italic`              | Is any text italic or oblique? |
| `underline`           | Underlined text indicator |
| `caps_ratio`          | % of capital letters in text |
| `word_count`          | Number of words |
| `title_case_ratio`    | % of words in Title Case |
| `has_numbering`       | Starts with number or bullet |
| `punctuation`         | Contains punctuation characters |
| `whitespace_above`    | Vertical whitespace from previous block |
| `whitespace_below`    | Vertical whitespace to next block |
| `bbox_x1/y1/x2/y2`    | Bounding box coordinates |
| `bbox_width/height`   | Width and height of bounding box |
| `relative_position`   | Normalized vertical location in page |
| `position_in_page`    | Position bin (e.g., "30%") |
| `alignment`           | Horizontal alignment: left/center/right |
| `lang`                | Language (default: English) |
| `page_number`         | Page index |
| `heading_level`       | Inferred from font size rank: H1/H2/H3 |
| `label`               | Original + inferred heading type |
| `FileID`              | PDF file identifier |

---

## 📤 Extraction Procedure (Step-by-Step)

1. For **DocBank**:
    - Load annotation JSONs
    - Filter and clean raw blocks
    - Merge blocks vertically (label-wise)
    - Estimate font size heuristically
    - Extract linguistic + spatial features
    - Save to CSV

2. For **DocLayNet**:
    - Open PDFs with `pdfplumber`
    - Merge words into line-level blocks
    - Infer labels (e.g., List, Caption) using font + text rules
    - Estimate font size and ranks
    - Extract same feature set
    - Save to CSV

3. Merge both datasets into a combined CSV:
    ```bash
    merged_features2.csv
    ```
