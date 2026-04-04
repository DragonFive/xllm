---
hide:
  - navigation
---
<style>
.md-content h1:first-of-type {
    display: none;
}
</style>


<div style="text-align:center">
    <img src="../assets/logo_with_llm.png" alt="xLLM" style="width:50%; height:auto;">
</div>

## Project Overview

**xLLM** is an efficient and user-friendly LLM intelligent inference framework that provides enterprise-level service guarantees and high-performance engine computing capabilities for model inference on domestic AI accelerators.


#### Background

LLMs with parameter counts ranging from tens of billions to trillions are being rapidly deployed in core business scenarios such as intelligent customer service, real-time recommendation, and content generation. Efficient support for domestic computing hardware has become a core requirement for low-cost inference deployment. Existing inference engines still struggle to adapt to the architectural characteristics of dedicated accelerators such as domestic chips. Problems such as low utilization of compute units, load imbalance and communication bottlenecks under the MoE architecture, and difficulties in KV cache management restrict both inference efficiency and system scalability. The xLLM inference engine improves resource efficiency across the entire communication-computation-storage path and provides critical technical support for large-scale LLM deployment in real-world business scenarios.


## Core Features

**xLLM** delivers robust intelligent computing capabilities. By leveraging hardware system optimization and algorithm-driven decision control, it jointly accelerates the inference process, enabling high-throughput, low-latency distributed inference services.

**Full Graph Pipeline Execution Orchestration**
- Asynchronous decoupled scheduling at the requests scheduling layer, to reduce computational bubbles.
- Asynchronous parallelism of computation and communication at the model graph layer, overlapping computation and communication.
- Pipelining of heterogeneous computing units at the operator kernel layer, overlapping computation and memory access.

**Graph Optimization for Dynamic Shapes**
- Dynamic shape adaptation based on parameterization and multi-graph caching methods to enhance the flexibility of static graph.
- Controlled tensor memory pool to ensure address security and reusability.
- Integration and adaptation of performance-critical custom operators (e.g., *PageAttention*, *AllReduce*).

**MoE Kernel Optimization**
- *GroupMatmul* optimization to improve computational efficiency.
- Chunked Prefill optimization to support long-sequence inputs.

**Efficient Memory Optimization**
- Mapping management between discrete physical memory and continuous virtual memory.
- On-demand memory allocation to reduce memory fragmentation.
- Intelligent scheduling of memory pages to increase memory reusability.
- Adaptation of corresponding operators for domestic accelerators.

**Global KV Cache Management**
- Intelligent offloading and prefetching of KV in hierarchical caches.
- KV cache-centric distributed storage architecture.
- Intelligent KV routing among computing nodes.

**Algorithm-driven Acceleration**
- Speculative decoding optimization to improve efficiency through multi-core parallelism.
- Dynamic load balancing of MoE experts to achieve efficient adjustment of expert distribution.

## Design Documents

- [Graph Mode Design Document](design/graph_mode_design.md)
- [Generative Recommendation Design Document](design/generative_recommendation_design.md)
