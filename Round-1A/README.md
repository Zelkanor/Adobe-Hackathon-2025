-----

# 📄 Adobe Hackathon 2025 — Round 1A

## Structured Outline Extractor

### 🔍 Problem Statement

Given a PDF document, the mission is to extract its structured outline—including the document title and all internal headings (H1, H2, H3) with their corresponding page numbers—and output this information in a standardized JSON format.

**Example Output:**

```json
{
  "title": "Annual Business Report",
  "outline": [
    { "level": "H1", "text": "Executive Summary", "page": 2 },
    { "level": "H2", "text": "Financial Highlights", "page": 3 }
  ]
}
```

Our solution is a hybrid approach, combining rule-based heuristics with a trained machine learning model to ensure both high precision and robust adaptability across diverse document types.

-----

### ⚙️ Our Approach

Our system is divided into two distinct pipelines: one for training the heading detection model and one for performing inference on new documents.

#### Training Strategy (`training.py`)

The training pipeline is designed to create a high-quality dataset and train a compact, efficient classifier.

  * **Phase 1: Data Preparation**

      * **Source Datasets**: We leveraged the **DocLayNet** and **DocBank** datasets for their rich layout annotations and high-quality labels.
      * **Feature Engineering**: A custom feature extraction engine (`common_features.py`) was used to create a shared schema, producing a vector of text and layout features for each text block (e.g., font size, alignment, caps ratio, vertical spacing).
      * **Data Cleaning**: A title-detection heuristic (largest, topmost text on the first page) was used to remove document titles from the training data, preventing confusion with H1 headings. Only relevant heading blocks (H1, H2, H3) and paragraphs were retained.
      * **Final Dataset**: The output, `cleaned_for_training.csv`, serves as the input for model training.

  * **Phase 2: Model Training with LightGBM**

      * **Model Choice**: We selected a **LightGBM** multiclass classifier due to its exceptional performance with a mix of tabular and sparse data. It was trained on:
          * Numerical, boolean, and categorical layout features.
          * Text features vectorized using character-level **TF-IDF** (n-grams from 2 to 5).
      * **Key Advantages**: LightGBM is highly efficient, supports incremental training (allowing us to process large datasets in chunks), and produces a highly compact (\<10MB) and interpretable final model.
      * **Model Artifacts**: The final trained components are saved to the `final_heading_model/` directory:
          * `lgbm_model.txt`
          * `tfidf_vectorizer.joblib`
          * Feature schemas and a `label_map.joblib`

#### Prediction Pipeline (`predict.py`)

The inference pipeline is optimized for fast, offline execution within a Docker container.

1.  **PDF Parsing**: **PyMuPDF** extracts text blocks and their rich metadata (bounding boxes, font info, etc.). Feature values are then computed to match the training schema.
2.  **Title Detection**: A heuristic is first applied to identify and extract the document's main title, which is then excluded from the heading prediction candidates.
3.  **Document Type Classification**: To handle edge cases, each PDF is classified as:
      * `standard` → The ML model is used for prediction.
      * `stylized` (e.g., flyers, posters) → A rule-based H1 detection is used.
      * `form` → Heading extraction is skipped as it's not applicable.
4.  **Feature Preparation**: Text features are vectorized using the saved TF-IDF model, and categorical features are one-hot encoded.
5.  **LightGBM Inference**: The trained model predicts a heading level for each text block. The final predictions are formatted into the required JSON structure.

-----

### 📂 Directory Structure

```
.
├── final_heading_model/  # Saved model artifacts
├── training/training.py  # Model training script
├── input/                # PDF input folder (mounted in Docker)
├── output/               # Output JSONs are written here
├── Dockerfile
├── predict.py            # The main inference script for the container

```

-----

### 🐳 Docker Build & Run Instructions

To build and run this solution in the evaluation environment, follow these steps.

1.  **Build the Docker Image**

    ```bash
    docker build --platform=linux/amd64 -t mysolutionname:somerandomidentifier .
    ```

2.  **Prepare Input & Output Folders**
    Ensure your local directory contains:

      * An `input/` folder with the `.pdf` files to be processed.
      * An `output/` folder (can be empty) where the results will be saved.

3.  **Run the Container**

    ```bash
    docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
    ```

4.  **Verify Output**
    For every `filename.pdf` in the `input` folder, a corresponding `filename.json` will be created in the `output` folder.

-----

### 🧠 Key Advantages of This Approach

  * **Hybrid Power**: Rules handle predictable structures (titles, forms), while the ML model handles nuanced layouts that rules can't generalize.
  * **Efficient & Compact**: The entire solution is fully offline. The LightGBM model is under 10MB, and the dependency footprint is minimized for a small Docker image.
  * **Robust**: The document type classification prevents the ML model from making poor predictions on unconventional layouts like forms or posters.
  * **Modular**: Title detection, document classification, and heading prediction are independent components that can be tuned or swapped.