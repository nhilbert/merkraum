# EpistBench — Epistemological Memory Benchmark

*Measuring whether AI memory systems can know responsibly — not just remember.*

## Why Another Benchmark?

[LongMemEval](https://arxiv.org/abs/2410.10813) (ICLR 2025) is the standard agent memory benchmark: 500 questions testing retrieval and reasoning over long conversation histories. It answers: **can your agent remember?**

In regulated environments — finance, insurance, healthcare, public sector — remembering is necessary but insufficient. The EU AI Act (Art. 12-15) requires transparency and traceability. The Product Liability Directive (PLD 2024/2853) places AI software under strict liability. GDPR demands the right to deletion.

These regulations don't ask "can your agent retrieve the right document?" They ask: **can you prove what it knew, when it changed its mind, and why?**

EpistBench tests for these capabilities. It is complementary to LongMemEval, not competing.

## Five Dimensions

| # | Dimension | What It Tests |
|---|-----------|--------------|
| 1 | **Contradiction Detection** | When new information contradicts stored beliefs, does the system detect and surface the conflict? |
| 2 | **Belief Lifecycle Tracking** | Can the system narrate how beliefs evolved over time — formation, confidence changes, supersession, retirement? |
| 3 | **Provenance & Audit Trail** | Given a current belief, can the system trace it back to its source(s)? |
| 4 | **Multi-Source Conflict Resolution** | When multiple sources provide conflicting information, does the system detect and resolve the conflict? |
| 5 | **Regulatory Query Coverage** | Can the system answer the structured meta-queries that compliance officers and auditors actually ask? |

Each dimension uses rubric-based scoring with an LLM judge (GPT-4o), evaluating three metrics per dimension.

## Baseline Results: Plain LLM (No Memory System)

The baseline adapter has **no memory system**. It concatenates all ingested documents into a context window and answers queries via GPT-4o-mini. This establishes the floor — what you get from a capable LLM without any structured knowledge management.

| Dimension | Score | Key Strength | Key Weakness |
|-----------|------:|:-------------|:-------------|
| Belief Lifecycle | **0.843** | Evolution narratives (0.90) | Temporal accuracy (0.83) |
| Regulatory Query | **0.817** | Queryability (0.95) | Completeness (0.68) |
| Contradiction Detection | **0.793** | Provenance (0.93) | Specificity (0.66) |
| Provenance & Audit | **0.580** | Source attribution (0.79) | Contradiction provenance (0.35) |
| Multi-Source Conflict | **0.383** | Authority awareness (0.60) | Resolution quality (0.20) |
| **Grand Mean** | **0.683** | | |

### Where the Baseline Is Strong (>0.75)

**Belief Lifecycle (0.843)**: LLMs excel at narrative reconstruction. When all information is in the context window, GPT-4o-mini tells coherent stories about belief evolution. But this degrades with context length — the baseline puts everything in one prompt, which works for a 100-scenario benchmark but would collapse at enterprise scale.

**Regulatory Query (0.817)**: High queryability (0.95) — the LLM produces structured, parseable answers. But completeness lags (0.68) — it misses regulatory details that a structured system would surface consistently. Accuracy (0.82) is solid for simple queries but drops on multi-regulation scenarios.

**Contradiction Detection (0.793)**: The LLM identifies direct contradictions when both facts are in the context window. But specificity (0.66) reveals the weakness: the LLM says "something changed" rather than precisely identifying which beliefs conflict.

### Where the Baseline Fails (<0.60)

**Provenance (0.580)**: Source attribution works when recent (0.79) but contradiction provenance collapses (0.35). When asked "what was the source of the original belief that was later contradicted?", the baseline has no structured provenance chain — just a flat list of documents.

**Multi-Source Conflict Resolution (0.383)**: The baseline's worst dimension. It detects authority (0.60) but cannot resolve conflicts (0.20). When multiple sources disagree, the LLM either picks one arbitrarily or hedges. There is no mechanism to weigh sources by authority, recency, or domain expertise. **This is where structured knowledge management provides the clearest advantage.**

## What This Means

The baseline (0.683) is the floor. Any memory system worth using should beat it — otherwise, concatenating documents into a context window is simpler and cheaper.

**Expected advantages of structured systems**:

| Capability | Baseline | Target for Structured System | Why |
|-----------|------:|------:|:------|
| Contradiction specificity | 0.66 | 0.85+ | Graph-based systems model beliefs as nodes with temporal metadata — they track state transitions structurally |
| Multi-source resolution | 0.20 | 0.60+ | Systems with weighted provenance can assess source authority, not just detect disagreement |
| Contradiction provenance | 0.35 | 0.75+ | Any system that maintains source attribution chains should beat flat document concatenation |
| Regulatory completeness | 0.68 | 0.85+ | Structured schemas with compliance-oriented indexing surface details that free-text retrieval misses |

**Where baselines are hard to beat**:
- Lifecycle narratives (0.90) — LLMs are inherently strong at narrative generation
- Queryability (0.95) — LLMs produce well-formatted answers natively

## Methodology

- **100 scenarios** across 5 dimensions (20 each), difficulty-balanced
- **Adapter architecture**: Common interface (`ingest`, `query`, `get_beliefs`, `get_contradictions`, `get_provenance`) — implement one class to test any memory system
- **Judge**: GPT-4o with dimension-specific rubrics (3 metrics per dimension, 0-1 scale)
- **Domains**: Healthcare, finance, manufacturing, defense, telecommunications, ESG, regulatory compliance, HR, supply chain, construction
- **Scenarios**: Self-contained YAML files with ingestion sequences, evaluation queries, and expected results

## Adapters

| Adapter | Type | Status |
|---------|------|--------|
| **baseline** | Plain LLM (GPT-4o-mini), no memory | Baseline results available |
| **merkraum** | Fixed-schema Neo4j with belief tracking | In progress |
| **mem0** | Open-vocabulary vector store | In progress |

Adding a new adapter: implement the `MemoryAdapter` abstract class (6 methods + name property).

## Open Source

EpistBench scenario files, adapter interface, evaluation harness, and judge rubrics are available for reproduction and extension. We welcome community contributions — additional scenarios, new adapters, and peer review of the rubrics.

## Honest Limitations

- **Judge reliability**: LLM-as-judge (GPT-4o) introduces evaluation variance. We chose rubric-based scoring to reduce subjectivity, but judge agreement studies are pending.
- **Scenario coverage**: 100 scenarios across 10 domains is a starting point, not comprehensive. Enterprise deployments involve domain-specific edge cases these scenarios may not cover.
- **Scale**: Scenarios are designed for benchmark-scale ingestion (5-20 documents per scenario). Real-world memory systems handle thousands to millions of entries — EpistBench does not test at production scale.
- **Cost**: Running the full benchmark with the LLM judge costs approximately $15-25 in API fees per adapter.

---

*EpistBench is developed by [merkraum](https://merkraum.de) as an open benchmark for epistemological capabilities in AI memory systems.*
