-----

# JITAR: Persona-Driven Document Intelligence

### A Fully Offline, Adaptive RAG Pipeline for CPU-Based Document Analysis

JITAR is a next-generation document analysis pipeline, purpose-built for on-device intelligence under strict CPU and model size constraints. It transforms static PDFs into dynamic sources of insight by combining modern layout analysis, hybrid retrieval, and adaptive reasoning. The system directly prioritizes and extracts information that matters to a specific user **persona** and their **job-to-be-done**, all while running completely offline.

-----

### ⚙️ The JITAR Pipeline: A Step-by-Step Breakdown

Our system processes documents through an advanced, multi-stage pipeline to ensure the highest relevance and quality.

#### 1\. 📄 Layout-Aware Parsing

Instead of naive text splitting, JITAR first uses the `unstructured.io` library with a vision-based model (like YOLOvX) to perform semantic chunking. It identifies the true layout of each page—including titles, paragraphs, lists, and tables—to create contextually whole and meaningful chunks. This robust first step is critical for the accuracy of the entire pipeline.

#### 2\. 🔍 Hybrid Retrieval

To ensure no relevant information is missed, we use a two-pronged retrieval strategy:

  * **Dense (Semantic) Search**: A `sentence-transformers` model with a `FAISS` index finds chunks that are contextually and conceptually similar to the user's query.
  * **Sparse (Lexical) Search**: A `BM25` index finds chunks that contain exact keyword matches.

The combination of these two methods guarantees high recall, creating a comprehensive list of candidate chunks.

#### 3\. 🎯 High-Precision Reranking

The large list of candidates is then passed to a powerful `CrossEncoder` reranking model (`ms-marco-TinyBERT-L-2-v2`). By analyzing the query and each chunk simultaneously, this model provides a highly accurate relevance score, ensuring only the absolute best chunks proceed to the next stage.

#### 4\. 🧠 Adaptive Strategy Selection

This is JITAR's key innovation. Before the final selection, a keyword-based heuristic analyzes the user's "job-to-be-done":

  * If the task is **exploratory** (e.g., "plan a trip"), the pipeline enforces **document diversity**, ensuring the top results come from multiple source files to provide a broad perspective.
  * If the task is **factual** (e.g., "how to create a form"), the pipeline prioritizes **precision**, selecting the best-ranked chunks regardless of their source.

#### 5\. ✍️ Persona-Driven Generation

The final, ranked, and diversified list of chunks is passed to a Small Language Model (`Qwen2-1.5B GGUF`). Guided by a persona-aware prompt, the SLM performs two tasks:

1.  **Title Generation**: It creates a concise, thematic title for each of the top sections.
2.  **Text Extraction**: It extracts only the most important sentences and instructions from each chunk that are directly relevant to the persona's goal.

-----

### 🚀 Core Technologies

  * **Parsing**: `unstructured` library with local vision models
  * **Retrieval**: `sentence-transformers`, `FAISS`, `rank_bm25`
  * **Reranking**: `sentence-transformers CrossEncoder`
  * **Generation**: `Qwen2-1.5B (GGUF)` via `llama-cpp-python`
  * **Orchestration**: `Python`, `Docker`

-----

### 🐳 Build & Run Instructions

1.  **Build the Docker Image**
    *Ensure Docker is running. In your project's root directory, execute:*

    ```bash
    docker build --platform linux/amd64 -t jitar_hackathon .
    ```

2.  **Prepare Input & Output Folders**
    *Create two folders in your project directory:*

      * `input/`: Place your request JSON file and all referenced PDF documents here.
      * `output/`: This folder will be created if it doesn't exist and will store the final `output.json`.

3.  **Run the Container**
    *Execute the following command to run the pipeline:*

    ```bash
    docker run --rm \
      -v "$(pwd)/input:/app/input" \
      -v "$(pwd)/output:/app/output" \
      --network none \
      jitar_hackathon
    ```

-----

### ✨ Key Advantages

  * **Fully Offline & CPU-Native**: All components, from parsing to generation, run without an internet connection on standard CPU hardware.
  * **Adaptive Intelligence**: The pipeline dynamically chooses between a "precision" or "diversity" strategy based on the user's task, ensuring optimal results for any scenario.
  * **Layout-Aware & Robust**: By understanding the visual layout of documents, JITAR avoids the errors common in simple text-based systems and handles complex PDFs with ease.
  * **Resource-Efficient**: The entire model stack is carefully selected to stay under the 1GB limit, and the pipeline is optimized to deliver results in under 60 seconds.