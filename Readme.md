# Semantic Quote Search using Retrieval-Augmented Generation (RAG) Pipeline

## 1. Introduction

This project presents an intelligent and interactive system for searching English quotes using a Retrieval-Augmented Generation (RAG)-inspired architecture. The system is designed to process a large dataset of quotes, encode the information semantically using sentence embeddings, index it with FAISS for efficient similarity search, and serve search results through a Gradio-based user interface. This allows users to input natural language queries and receive the most relevant quotes based on semantic similarity, not just keyword matching.

The system is implemented in modular steps, suitable for Jupyter, Colab, or standalone Python scripts. It also provides a fallback sample dataset and various robustness checks, ensuring operability even in constrained environments.

---

## 2. Objectives

* Build a scalable and interactive quote retrieval system.
* Use state-of-the-art sentence embedding models.
* Index and search with FAISS for fast similarity retrieval.
* Design a clean and intuitive Gradio interface.
* Provide fallback mechanisms for offline or restricted execution.
* Allow easy customization, testing, and deployment.

---

## 3. System Architecture

### 3.1 Overview

The system is structured into the following main components:

1. **Package Installation**
2. **Library Imports and Setup**
3. **Data Processing Class**
4. **Embedding Model Class**
5. **RAG Pipeline Class**
6. **System Initialization**
7. **Test Search Function**
8. **Gradio Interface Creation**
9. **Interface Launch**
10. **Cleanup and Stop Functions**

Each component is modular, enabling flexible modification, debugging, and expansion.

---

## 4. Implementation Details

### 4.1 Cell 1 - Package Installation

To ensure compatibility across environments like Colab and Kaggle, the following packages are installed:

```python
!pip install -q sentence-transformers datasets transformers torch torchvision torchaudio
!pip install -q faiss-cpu pandas numpy scikit-learn
!pip install -q gradio
!pip install -q huggingface_hub accelerate fsspec
```

These include core packages for data processing, deep learning, semantic search, and UI rendering.

---

### 4.2 Cell 2 - Import Libraries and Setup

All relevant libraries are imported:

* `torch`, `numpy`, `pandas` for data and computation
* `sentence_transformers` for model embedding
* `faiss` for similarity search
* `datasets` for loading Hugging Face datasets
* `gradio` for UI

The device (GPU/CPU) is detected using PyTorch for optimized performance.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

### 4.3 Cell 3 - Data Processing Class

`ColabQuoteDataProcessor` handles:

* **Data Loading:**

  * Tries to read the JSONL dataset from Hugging Face.
  * If fails, falls back to `load_dataset()`.
  * If both fail, creates a small sample dataset.

* **Preprocessing:**

  * Removes rows with missing values.
  * Standardizes quote and author text.
  * Parses and formats `tags` column.
  * Creates a `search_text` field combining quote, author, and tags.

Sample fallback dataset contains 10 popular quotes for demo or offline use.

---

### 4.4 Cell 4 - Embedding Model Class

`ColabQuoteEmbeddingModel` initializes the sentence transformer `all-MiniLM-L6-v2`, a lightweight yet effective transformer model.

* Automatically loads to CUDA if available, otherwise uses CPU.
* Encodes queries and dataset entries into 384-dimensional dense vectors.

---

### 4.5 Cell 5 - RAG Pipeline Class

`ColabQuoteRAGPipeline` handles the core functionality:

* **Embedding Creation:**

  * Batches and encodes all `search_text` entries.
  * Normalizes vectors and builds a FAISS `IndexFlatIP` for inner-product similarity.

* **Quote Retrieval:**

  * Encodes input query.
  * Performs top-K similarity search.
  * Returns ranked, formatted results with similarity scores.

---

### 4.6 Cell 6 - System Initialization

This cell orchestrates:

* Dataset loading (`max_samples=1000` for Colab efficiency)
* Data preprocessing
* Embedding model loading
* RAG pipeline setup
* Embedding creation with time tracking

Final objects are stored globally for reuse:

```python
rag_system = rag_pipeline
```

---

### 4.7 Cell 7 - Test Search Function

A built-in function `quick_test()` is defined to test the full pipeline using a query like `motivation`. It checks:

* Query encoding
* FAISS search
* Result formatting

A second function `search_quotes()` is built for Gradio integration. It handles:

* Empty queries
* Valid query processing
* Result formatting with similarity, author, tags

---

### 4.8 Cell 8 - Gradio Interface Creation

The `create_interface()` function sets up a clean user interface using `gr.Blocks`:

* **Input Controls:**

  * Textbox for user queries
  * Slider for number of results (1-10)

* **Results Area:**

  * Markdown component for displaying formatted output

* **Example Buttons:**

  * Quick-access buttons for common queries like "motivational quotes" or "Steve Jobs"

User actions (button clicks) are connected to `search_quotes()`.

---

### 4.9 Cell 9 - Launch Interface

The interface is launched with public sharing using:

```python
demo.launch(share=True, debug=True, height=600, show_error=True, quiet=False)
```

This creates a public link, allowing anyone to use the interface without needing a local setup.

---

### 4.10 Cell 10 - Stop and Cleanup

Cleans up resources:

* Closes Gradio instance
* Deletes large objects (`rag_system`, `demo`, etc.)
* Triggers garbage collection

This step is useful before restarting or switching to another project in the same Colab kernel.

---

## 5. Evaluation and Results

### 5.1 Functional Evaluation

The system was tested with several queries:

* **"quotes about life"**
* **"Steve Jobs quotes"**
* **"inspirational quotes"**

Results were:

* Fast (response in under 1s for small datasets)
* Relevant (top matches contextually appropriate)
* Intuitive (tags and authors displayed clearly)

### 5.2 Robustness

* Handles missing datasets gracefully
* Works on both CPU and GPU
* Ensures consistent results via sampling (`random_state=42`)

---

## 6. Design Decisions

* **MiniLM over BERT:** Chosen for its balance between speed and semantic quality.
* **FAISS IndexFlatIP:** Simple inner-product index with L2 normalization to simulate cosine similarity.
* **Gradio UI:** Easier deployment and sharing compared to Flask or Streamlit.
* **Modularity:** Every component is a class or function for clarity, testing, and reuse.
* **Fallback sample data:** Ensures demo readiness even without internet.

---

## 7. Challenges Faced

* Hugging Face’s new `hf://` format doesn’t work with `pd.read_json()` directly.
* Large embedding batches may cause OOM in restricted environments; addressed via batching.
* Ensuring tag consistency required type-checking and fallback formatting.
* FAISS indexing must be normalized to approximate cosine similarity.
* Gradio’s callback outputs required careful matching to input widget types.

---

## 8. Future Enhancements

* Add **author-based filtering** or dropdown selection
* Allow **tag-based refinement** or search-by-category
* Introduce **multilingual support**
* Upgrade to **retrieval + generation model** for full RAG (e.g., T5 with context)
* Add **embedding cache** and **quantized FAISS index** for large datasets
* Integrate with **Streamlit** or **Hugging Face Spaces** for deployment

---

## 9. Running the Project

### 9.1 Prerequisites

* Python >= 3.7
* Internet connection (for loading model/dataset)

### 9.2 Run Locally or in Colab

1. Install packages (Cell 1)
2. Run each subsequent cell in order (2 to 10)
3. Open the Gradio URL (after Cell 9)
4. Use sample queries or enter your own

### 9.3 Run from Script

Convert the notebook cells into Python files:

```bash
python main.py  # Contains combined logic
python app.py   # Only launches Gradio interface
```

---

## 10. Conclusion

This project effectively demonstrates a light Retrieval-Augmented Generation pipeline applied to quote search. Leveraging MiniLM embeddings, FAISS vector indexing, and Gradio UI, it provides an efficient and user-friendly semantic search system. The modular design allows flexibility in extension, experimentation, and deployment.

By combining strong NLP tools with practical engineering, the system serves as a foundational framework for broader document or FAQ retrieval use cases, and can be further enhanced for multi-modal or generative applications.

---

## 11. References

* Hugging Face Datasets: [https://huggingface.co/datasets](https://huggingface.co/datasets)
* Sentence Transformers: [https://www.sbert.net/](https://www.sbert.net/)
* FAISS: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
* Gradio: [https://gradio.app/](https://gradio.app/)
* Dataset Source: [https://huggingface.co/datasets/Abirate/english\_quotes](https://huggingface.co/datasets/Abirate/english_quotes)

---

## 12. License

This project is open for non-commercial, educational, and research purposes. All models and datasets are under their respective licenses as per Hugging Face and original authors.

