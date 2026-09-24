# arXiv Dense Retrieval with BGE and FAISS

<img width="665" height="300" alt="Latency profiling of the dense-retrieval baseline" src="https://github.com/user-attachments/assets/40492315-0455-4b21-80fe-32aa0e538ae9" />

A documented Google Colab experiment in **dense document retrieval** over arXiv paper titles and abstracts. Given an English natural-language query, the system returns ranked paper identifiers, titles, and cosine-similarity scores. It uses `BAAI/bge-base-en-v1.5` for embeddings and an exact FAISS index for retrieval. [1] [2]

This project is intentionally limited to retrieval. It does not implement question answering, retrieval-augmented generation, scientific advice generation, a recommendation system, or a reranking stage.

## Final results

The final CLS-pooling configuration was evaluated on **1,000 labelled queries** over **98,213 indexed paper records**. The target for the coursework task was `MRR@5 > 0.91`.

| Measure | Result |
| --- | ---: |
| **MRR@5** | **0.9160** |
| Coursework target | **Met** |
| Document-embedding time | 560.5 seconds |
| Embedding dimension | 768 |
| Documents truncated at 512 tokens | 792 / 98,213 (**0.81%**) |
| Longest evaluation query | 78 tokens |

The latency benchmark used a 50-query warm-up. PyTorch had access to an NVIDIA A100-SXM4-40GB GPU for query embedding, while FAISS `IndexFlatIP` ran on CPU.

| Retrieval stage | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: |
| Query embedding | 9.85 | 9.79 | 10.39 |
| Exact FAISS search | 11.97 | 9.41 | 23.18 |
| Result materialisation | 0.62 | 0.61 | 0.65 |
| **End-to-end total** | **22.44** | **19.87** | **33.69** |

Observed sequential throughput was **44.51 queries per second**. In this run, FAISS search accounted for 53.4% of mean component time, query embedding for 43.9%, and result materialisation for 2.7%. The Pearson correlation between query length in characters and total latency was -0.014, which indicates no meaningful linear association in this sample.

> These are observed sequential single-query results from the recorded Google Colab runtime. They are not production service-level guarantees or general benchmarks for search over the full arXiv corpus.

## Method

The notebook validates the input schema, checks paper identifier uniqueness, and confirms that every labelled target is present in the collection. It then concatenates each title and abstract into a document string.

`BAAI/bge-base-en-v1.5` creates the document and query embeddings. The implementation uses the model card's CLS-token representation, `last_hidden_state[:, 0]`, followed by L2 normalisation. The recommended BGE retrieval instruction is prepended to short queries only; documents do not receive it. [1] The normalised vectors are indexed with CPU-based `faiss.IndexFlatIP`, where inner-product ranking is equivalent to cosine-similarity ranking. [2]

For every evaluation query, the system retrieves five documents. **Mean Reciprocal Rank at 5** assigns the reciprocal of the rank of the known relevant document if it appears in the top five, otherwise zero, and averages this quantity over all queries.

## Data availability

The course archive is **not included and no public download link is provided**. It contains the arXiv-derived metadata and the course evaluation split, whose redistribution conditions have not been verified. The repository is therefore an open record of the implementation, experiment configuration, and final outputs, rather than a self-contained runnable demo.

The executed notebook retains its original Google Drive paths for authorised holders of the archive:

```python
ZIP_PATH = Path('/content/drive/MyDrive/nlp_s3_data.zip')
ARTIFACT_DIR = Path('/content/drive/MyDrive/arxiv_retrieval')
```

The expected archive structure is:

```text
nlp_s3_data.zip
└── nlp_s3_project/
    ├── arxiv-metadata-s.json
    └── test_sample.csv
```

No raw data, evaluation labels, embedding matrices, FAISS indexes, model cache, or credentials are committed. A future independently reproducible version should cite a precise public data release and its licence. arXiv documents official mechanisms for metadata access, but this repository does not claim that the course split itself is an official arXiv release. [3]

## Repository contents

```text
.
├── arxiv_dense_retrieval.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The notebook includes the executed final outputs, data checks, exploratory analysis, a token-length audit, a qualitative top-5 example, MRR@5 calculation, and component-level latency profiling.

## Scope and limitations

The project reports a **single-stage dense-retrieval baseline**. It does not empirically compare BGE with BM25, TF-IDF, alternative embedding models, approximate FAISS indexes, or cross-encoder rerankers. It should not be used to claim that dense retrieval is universally superior to sparse retrieval.

The index uses exact search. At a larger collection scale, approximate FAISS alternatives such as HNSW or IVF should be compared against this baseline using both retrieval quality and latency. A cross-encoder reranker may improve ranking quality but would introduce extra inference latency and requires a separate measured experiment.

## References

[1]: https://huggingface.co/BAAI/bge-base-en-v1.5 "BAAI/bge-base-en-v1.5 model card"
[2]: https://github.com/facebookresearch/faiss/wiki/Getting-started "FAISS: Getting started"
[3]: https://info.arxiv.org/help/bulk_data.html "arXiv bulk data access"
