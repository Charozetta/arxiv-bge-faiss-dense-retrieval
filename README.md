# arXiv Dense Retrieval with BGE & FAISS

<img width="665" height="300" alt="image" src="https://github.com/user-attachments/assets/40492315-0455-4b21-80fe-32aa0e538ae9" />


A reproducible Google Colab baseline for retrieving relevant arXiv papers from natural-language queries. The project uses the `BAAI/bge-base-en-v1.5` embedding model and an exact FAISS index to rank paper titles and abstracts by cosine similarity.

The notebook evaluates retrieval quality with **Mean Reciprocal Rank at 5 (MRR@5)** and profiles sequential query latency by separating embedding, FAISS search, and result post-processing.

## Overview

This project implements **document retrieval** over a fixed collection of arXiv papers. Given a natural-language query, it returns a ranked list of paper identifiers, titles, and similarity scores.

It does not implement:

- question answering;

- retrieval-augmented generation (RAG);

- paper recommendation;

- generated scientific advice;

- a reranking stage.

The intended use is to provide a transparent and reproducible dense-retrieval baseline.

## Method

The retrieval pipeline consists of four stages:

1. **Data validation and exploratory analysis.** The notebook verifies the input schema, checks that article identifiers are unique, and confirms that every evaluation target appears in the indexed collection.

1. **Dense embedding generation.** Paper titles and abstracts are concatenated and embedded with `BAAI/bge-base-en-v1.5`. Queries receive the BGE retrieval instruction:

   ```
   Represent this sentence for searching relevant passages:
   ```

1. **Exact similarity search.** Embeddings are L2-normalized and stored in a CPU-based `faiss.IndexFlatIP` index. For normalized vectors, inner-product ranking is equivalent to cosine-similarity ranking.

1. **Evaluation and profiling.** The notebook calculates MRR@5, prints a qualitative top-5 retrieval example, and measures sequential latency for query embedding, FAISS search, and result materialization.

## Model and index configuration

| Component | Configuration |
| --- | --- |
| Embedding model | `BAAI/bge-base-en-v1.5` |
| Embedding pooling | CLS token: `last_hidden_state[:, 0]` |
| Vector normalization | L2 normalization |
| Maximum input length | 512 tokens |
| Query instruction | BGE retrieval instruction |
| Vector index | `faiss.IndexFlatIP` |
| Search type | Exact CPU inner-product search |
| Evaluation metric | MRR@5 |
| Random seed | 42 |

> **Important:** The notebook uses CLS-token pooling. Any MRR@5 value must be generated from this exact configuration and must not be copied from an earlier experiment that used another pooling method.

## Input data

The input archive is intentionally excluded from version control. The notebook expects a ZIP archive with the following structure:

```
nlp_s3_data.zip
└── nlp_s3_project/
    ├── arxiv-metadata-s.json
    └── test_sample.csv
```

### `arxiv-metadata-s.json`

The metadata file must contain records with at least these fields:

| Field | Description |
| --- | --- |
| `id` | Unique arXiv paper identifier |
| `title` | Paper title |
| `abstract` | Paper abstract |

### `test_sample.csv`

The evaluation file must contain the following columns:

| Column | Description |
| --- | --- |
| `id` | Identifier of the known relevant paper |
| `query` | Natural-language search query |

An optional `abstract` column may be present for exploratory analysis.

## Running in Google Colab

The notebook is designed for Google Colab because the input archive and derived artifacts can be kept in Google Drive.

1. Upload `notebooks/arxiv_dense_retrieval.ipynb` to Google Colab.

1. Place the dataset archive in Google Drive.

1. Update `ZIP_PATH` in the notebook if the archive has a different name or location:

   ```python
   ZIP_PATH = Path('/content/drive/MyDrive/nlp_s3_data.zip')
   ```

1. Optionally update the artifact directory:

   ```python
   ARTIFACT_DIR = Path('/content/drive/MyDrive/arxiv_retrieval')
   ```

1. Run all cells from top to bottom.

The notebook installs the required retrieval packages in its first setup cell:

```python
%pip install -q -U "transformers>=4.45" "faiss-cpu>=1.8.0"
```

## Generated artifacts

The notebook writes reusable artifacts outside the Git repository:

```
arxiv_retrieval/
├── article_embeddings_cls.npy
├── faiss_index_cls.bin
└── bge_base_en_v15_cls_encoder.onnx   # optional
```

Set `REBUILD_INDEX = False` only when the embeddings and FAISS index were created from the same corpus and the same CLS-pooling configuration.

## Evaluation

For each evaluation query, the system retrieves the top five papers. Each query has one known relevant paper identifier.

The reported metric is:

$$
\mathrm{MRR@5} = \frac{1}{|Q|}\sum_{q \in Q}
\begin{cases}
\frac{1}{\operatorname{rank}_q}, & \text{if the relevant document is in the top 5} \\
0, & \text{otherwise}
\end{cases}
$$

The project target is:

```
MRR@5 > 0.91
```

After a clean run, record the following values in the repository or a GitHub Release:

- final MRR@5;

- number of indexed papers and evaluation queries;

- document truncation rate at 512 tokens;

- Python, PyTorch, Transformers, and FAISS versions;

- execution device and FAISS CPU-thread count;

- mean, median, and p95 sequential latency;

- a representative top-5 retrieval example.

## Performance measurement

Latency is measured sequentially for one query at a time after a warm-up phase. The notebook reports:

- query embedding time;

- FAISS search time;

- result post-processing time;

- total end-to-end latency;

- mean, median, and p95 latency;

- sequential throughput in queries per second.

The FAISS index runs on CPU. When CUDA is available, query embedding may run on GPU; therefore, the end-to-end timing includes the CPU/GPU transfer required before FAISS search.

> These measurements describe one Colab runtime and are not production service-level guarantees.

## Limitations

The project is a single-stage dense-retrieval baseline. It does not compare BGE against BM25, TF-IDF, alternative embedding models, approximate FAISS indexes, or cross-encoder rerankers.

The index uses exact search. At a larger corpus scale, `IndexHNSWFlat` or IVF-based FAISS indexes should be evaluated against the baseline while reporting both retrieval quality and latency.

The optional ONNX export is disabled by default. TensorRT compilation depends on a compatible CUDA and TensorRT environment and should be benchmarked separately from the native PyTorch baseline.

## Data and model usage

Do not commit the following files until their redistribution terms have been checked:

- the source ZIP archive;

- raw arXiv metadata;

- evaluation CSV files;

- cached model files;

- embeddings;

- FAISS indexes;

- ONNX exports.

The repository should contain code and documentation only unless the data license explicitly permits redistribution.
