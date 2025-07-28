📄 Adobe Hackathon 2025 — Round 1A: Structured Outline Extractor
================================================================

🔍 Problem Statement
--------------------

Given a PDF document, extract its structured outline — including the document title and all internal headings (H1, H2, H3) along with their page numbers — and return this in a standardized JSON format.

Example output:

{"title": "Annual Business Report","outline": \[{ "level": "H1", "text": "Executive Summary", "page": 2 },{ "level": "H2", "text": "Financial Highlights", "page": 3 }\]}

Our approach is a hybrid solution: we combine rule-based heuristics with a trained machine learning model for heading detection, ensuring both precision and adaptability across document types.

⚙️ Approach Overview
--------------------

### 🧠 Training Strategy

The training pipeline is implemented in training.py and is divided into two phases:

1.  Data Cleaning & Preprocessing
    

*   Source datasets: DocLayNet and DocBank were chosen for their diverse layout annotations and high-quality labels.
    
*   A shared schema was enforced using a custom feature extraction engine (common\_features.py), producing text + layout features per text block (font size, alignment, caps ratio, spacing, etc.).
    
*   Titles were removed using a heuristic: the largest, topmost text block on the first page was excluded to avoid confusion with H1 headings.
    
*   Only relevant heading blocks (H1, H2, H3) were retained.
    
*   Final output: cleaned\_for\_training.csv — used for model training.
    

1.  Model Training with LightGBM
    

We use a LightGBM multiclass classifier trained on:

*   Layout features (numerical, boolean, categorical)
    
*   Text features vectorized via TF-IDF (character n-grams, range 2–5)
    

Key benefits of LightGBM:

*   Efficient with tabular + sparse data
    
*   Incremental training support (chunk-wise model updates)
    
*   Highly interpretable and compact model (<10MB)
    

The model is trained in chunks to avoid memory overflow on large datasets. Artifacts saved to final\_heading\_model/:

*   lgbm\_model.txt
    
*   tfidf\_vectorizer.joblib
    
*   feature schemas (numerical, categorical, dummy columns)
    
*   label\_map.joblib
    

### 🔎 Prediction Strategy (predict.py)

predict.py implements the complete inference pipeline, optimized for Docker execution.

1.  PDF Parsing
    

*   PyMuPDF is used to extract text blocks and bounding boxes.
    
*   Feature values (layout + formatting) are computed to match training.
    

1.  Document Title Detection
    

*   Heuristic: select the topmost, largest, centered text block on the first page.
    
*   The title is extracted and excluded from heading candidates.
    

1.  Document Type Classification
    

We classify each input PDF into one of:

*   standard → uses ML model
    
*   stylized → uses rule-based H1 detection
    
*   form → skips heading extraction
    

This ensures robust performance across diverse formats like flyers and forms.

1.  Feature Preparation
    

*   TF-IDF is applied to the text
    
*   One-hot encoding is applied to alignment categories using saved dummy columns
    
*   Numerical + boolean features are normalized
    

1.  LightGBM Inference
    

*   The trained model predicts heading levels for each block (H1/H2/H3)
    
*   Predictions are mapped to JSON format with block text and page number
    

Final output is saved as filename.json in /app/output

📂 Directory Structure
----------------------

.├── input/ # PDF input folder (mounted in Docker)├── output/ # Output JSONs written here├── Dockerfile├── training.py # Model training script├── predict.py # PDF inference script├── common\_features.py # Feature extraction logic├── process\_docbank.py # Dataset preparation├── process\_doclaynet.py # Dataset preparation├── merge\_datasets.py # Merge DocLayNet + DocBank└── final\_heading\_model/ # Saved model artifacts

🐳 Docker Instructions
----------------------

To build and run your solution in Adobe’s evaluation environment, follow the exact instructions below:

1.  🏗️ Build the Docker Image
    

docker build --platform=linux/amd64 -t mysolutionname:somerandomidentifier .

1.  📁 Prepare Input & Output Folders
    

Ensure your directory contains:

*   input/ folder with .pdf files
    
*   output/ folder (can be empty, used to store results)
    

1.  ▶️ Run the Container
    

docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier

1.  ✅ Output
    

For every input.pdf, a corresponding input.json will be saved to output/

Example:

input/├── policy.pdfoutput/├── policy.json

Each JSON contains:

*   document title (automatically detected)
    
*   outline: list of { level, text, page }
    

🧠 Why This Approach Works
--------------------------

*   Rules handle structure where ML struggles (e.g., titles, forms)
    
*   ML handles nuanced layout/text variations that rules cannot generalize
    
*   LightGBM + char-level TF-IDF capture multilingual, multi-style document formats
    
*   Fully offline & efficient: Docker runs without internet, model is under 10MB
    
*   Modular: title detection, doc classification, and heading prediction are independently swappable