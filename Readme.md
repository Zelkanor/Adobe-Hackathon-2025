# Adobe India Hackathon 2025 – Connecting the Dots 🚀

Welcome to our submission for the **Adobe India Hackathon 2025**, themed **"Connecting the Dots"**.

---

## 🌐 Problem Statement (Hackathon Theme)

> **Rethink Reading. Rediscover Knowledge.**

In a world flooded with documents, how do we move from *more content* to *more context*? The challenge invites developers to reinvent the PDF—turning it into an intelligent, interactive experience that understands structure, connects ideas, and responds like a trusted research companion.

This hackathon spans two phases:
- **Round 1**: Build the brains — intelligent understanding and extraction.
- **Round 2**: Build the interface — a futuristic web reading experience using Adobe's PDF Embed API.

---

## 🧠 Round 1A: Understand Your Document

### 🔍 Problem Overview

You’re handed a PDF- but instead of just reading it, you must make sense of it **like a machine**. Your mission is to extract a structured **outline** of the document including:

- **Title**
- **Headings**: H1, H2, H3 (with level and page number)

### ✅ Solution Requirements

- Accepts a PDF file (≤ 50 pages)
- Outputs a structured `JSON` like:
  ```json
  {
    "title": "Understanding AI",
    "outline": [
      { "level": "H1", "text": "Introduction", "page": 1 },
      { "level": "H2", "text": "What is AI?", "page": 2 },
      { "level": "H3", "text": "History of AI", "page": 3 }
    ]
  }

### 🔒 Constraints
- Must run on CPU (amd64) within a Docker container.
- Works offline, within 10 seconds, with no internet access.
- Model size (if any) ≤ 200MB.

### 📎 Click here to view our Round-1A solution and approach:
👉 [Round 1A README](https://github.com/Zelkanor/Adobe-Hackathon-2025/blob/main/Round-1A/README.md)


## 👤 Round 1B: Persona-Driven Document Intelligence

### 🎯 Problem Overview

Build a system that analyzes a collection of PDFs and extracts the most relevant sections based on:
- A persona (e.g., Student, Analyst, Researcher)
- A job-to-be-done (e.g., literature review, summarization, exam prep)
- The solution must generalize across various document types (research papers, textbooks, financial reports)

### 📤 Expected Output (JSON):
- Input metadata (persona, job, document list)
- Extracted sections with:
    - Document name
    - Page number
    - Section title
    - Importance ranking
    - Sub-section level summaries/refinements

### 🔒 Constraints:
- Runs on CPU (≤ 1GB model size)
- Max processing time: 60 seconds
- No network access

### 📎 Click here to view our Round-1B solution and approach:
👉 [Round 1B README](https://github.com/Zelkanor/Adobe-Hackathon-2025/blob/main/Round-1B/README.md)

## License
This project is licensed under the MIT License.
