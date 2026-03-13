<h1 align="center">
    QChunker: Learning Question-Aware Text Chunking for Domain RAG via Multi-Agent Debate
</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2603.11650">
        <img alt="arXiv Paper" src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv">
    </a>
    <a href="https://opensource.org/license/apache-2-0">
        <img alt="Apache 2.0 License" src="https://img.shields.io/badge/License-Apache_2.0-4285f4.svg?logo=apache">
    </a>
</p>

This study introduces **QChunker**, an innovative text chunking framework based on multi-agent debate. It reshapes the chunking process from a traditional, isolated, and passive preprocessing step into an active and forward-looking process of deep understanding and knowledge reconstruction, aiming to address the performance bottlenecks faced by Retrieval-Augmented Generation (RAG) systems due to semantic fragmentation at the source.

### 🎯 Who Should Pay Attention to Our Work?

This study is primarily targeted at researchers and practitioners deeply engaged in the fields of **Information Retrieval, Natural Language Processing, and Knowledge Engineering**. Specifically, the following groups will derive significant insights from our work:

* **RAG System Developers and Optimization Engineers**: For engineers dedicated to enhancing the performance of RAG systems in handling specialized domain knowledge, this study uncovers the prevalent preprocessing bottleneck—semantic fragmentation caused by unreasonable text chunking—and offers a novel framework, QChunker, to address this issue at its root.
* **Researchers in Large Language Model (LLM) Applications**: Researchers focusing on applying LLMs to specific vertical domains (such as finance, law, healthcare, chemical safety, etc.) will find that our work provides crucial methodologies for constructing high-quality, model-friendly domain knowledge bases. It tackles the core challenge of effectively aligning external knowledge with the internal reasoning mechanisms of LLMs.
* **Knowledge Base Construction and Data Scientists**: Experts responsible for building enterprise-level or domain-specific knowledge bases have long faced the challenge of transforming unstructured documents into atomic, information-complete knowledge units. QChunker offers an automated solution driven by deep understanding, surpassing traditional heuristic rules.
* **NLP Evaluation and Benchmark Researchers**: The downstream-task-independent direct evaluation metric, ChunkScore, proposed in this study, opens up a new path for assessing text chunking quality, breaking free from the existing evaluation challenges of long chains, high costs, and difficult attribution. Meanwhile, the hazardous chemical safety dataset we constructed also provides the community with a high-quality evaluation benchmark.

## ✨ Core Contributions

The core contribution of this study lies in not only identifying and providing an in-depth analysis of a neglected performance bottleneck in current RAG systems but also proposing a systematic and complete solution from theory to practice. The specific contributions can be summarized as the following four points:

1. **Proposing an Innovative Text Chunking Framework (QChunker)**: We pioneer a text chunking framework based on multi-agent debate—QChunker. The core idea of this framework is to elevate text chunking from a passive "text segmentation" task to an active "knowledge reconstruction" process. By integrating text segmentation with knowledge completion, QChunker aims to generate logically independent and informationally complete knowledge units, fundamentally addressing the RAG performance degradation caused by semantic fragmentation.
2. **Designing an Efficient Direct Evaluation Metric (ChunkScore)**: To address the drawback of existing chunking evaluation methods that overly rely on downstream tasks, we design and validate a novel direct evaluation metric—ChunkScore. This metric innovatively quantifies chunking quality from two dimensions: "micro-level logical independence" and "macro-level semantic dispersion," enabling efficient, accurate, and independent judgment of chunking schemes. The metric not only serves as an evaluation tool but is also integrated as a key decision-making mechanism within the QChunker framework.
3. **Constructing and Open-Sourcing High-Quality Datasets**: To advance research in domain knowledge processing, we construct and contribute two valuable resources to the community: a general-purpose QChunker dataset containing 45,000 high-quality samples for training chunking models, and a specialized question-answering dataset focused on hazardous chemical safety, providing a solid evaluation benchmark for RAG research in specific domains.
4. **Providing Comprehensive Experimental Validation**: We systematically and multi-dimensionally evaluate the effectiveness of the QChunker framework through benchmark tests across four different domains. The experimental results convincingly demonstrate the significant superiority and immense potential of our method compared to existing technologies when empowering RAG systems to serve specific knowledge domains.

## **🛠️ Quick Start**

- Install dependency packages

```bash
pip install -r requirements.txt
```

- Start the milvus-lite service (vector database)

```bash
milvus-server --data /Storage/path/of/the/database
```

- Download models to corresponding directories.
- Modify various configurations  according to your need.
- Run `chunk_*.py` to accomplish the text chunking task for domain documents.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python chunk_gpt.py >> multifiled/qwen3_14B_set.log 2>&1 &
```

- Subsequently, execute  `quick_start.py` and `retrieval.py` to carry out the retrieval and question-answering processes.

```bash
CUDA_VISIBLE_DEVICES=1 nohup python quick_start.py 
--docs_path 'crud_qwen3_14B_set.json' 
--collection_name 'crud_qwen3_14B_set' 
--retrieve_top_k 8 
--task 'quest_answer' 
--construct_index 
>> log/qchunker_crud_qwen3_14B_set.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python retrieval.py 
--data_path 'evaldata/huagong_test.json'
--save_file 'eval/qchunker_huagong_qwen3_14B_set.json'
--docs_path 'huagong_qwen3_14B_set.json' 
--collection_name 'huagong_qwen3_14B_set' 
--retrieve_top_k 8 
--construct_index 
>> log/qchunker_huagong_qwen3_14B_set.log 2>&1 &
```

- Open and run `chunk.ipynb`, which will conduct a comprehensive quality assessment of the results generated by different chunking strategies.

### 📊 Results

Through comprehensive experimental evaluations, this study achieves a series of convincing results, clearly demonstrating the effectiveness and advancement of the QChunker framework and its core concepts:

* **Exceptional Performance of the QChunker Framework**: In benchmark tests across four heterogeneous domains, RAG systems employing QChunker for text chunking significantly outperformed traditional chunking methods in downstream question-answering tasks. This proves that our framework can effectively alleviate semantic fragmentation and provide higher-quality and more accessible knowledge contexts for large language models.

* **Effectiveness and Consistency of the ChunkScore Metric**: Both experimental and theoretical analyses validate the reliability of ChunkScore as a direct evaluation metric. Its scoring results exhibit a high positive correlation with the final performance of downstream RAG tasks, indicating that ChunkScore can accurately predict the quality of chunking schemes and successfully eliminate reliance on expensive and time-consuming end-to-end evaluations.

* **Effectiveness in Domain Knowledge Base Construction**: On our highly specialized dataset for hazardous chemical safety, QChunker demonstrates strong domain adaptability. The chunks it generated better preserved term definitions, complex causal relationships, and contextual dependencies, significantly enhancing the RAG system's ability to handle complex professional problems in vertical domains.

* **Potential for Methodological Universality**: The successful cross-domain experiments indicate that the concepts advocated by QChunker, such as "prioritizing deep understanding" and "balancing segmentation and completion," possess good universality, providing a promising new paradigm for addressing the processing and utilization of various knowledge-intensive documents.



























