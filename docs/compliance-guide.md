# Compliance Guide

*How Merkraum supports regulatory requirements for AI systems in the EU.*

## Regulatory Context

Three regulations are reshaping how organizations deploy AI in the EU:

| Regulation | Key Date | What It Requires |
|-----------|----------|-----------------|
| **EU AI Act** | Aug 2, 2026 (full enforcement) | Transparency, traceability, and human oversight for high-risk AI systems |
| **Product Liability Directive (PLD)** | Dec 9, 2026 (transposition) | AI software under strict product liability — burden of proof favors plaintiffs |
| **GDPR** | In force | Right to deletion (Art. 17), data minimization, purpose limitation |

For organizations deploying AI agents that make or inform decisions, the compliance question is no longer abstract. It's operational: **can you prove what your AI knew, when it learned it, and how it changed its mind?**

## How Merkraum Addresses Each Regulation

### EU AI Act — Transparency and Traceability

The AI Act's requirements for high-risk systems (Article 13, 14, 15) include:

- **Transparency**: Users must be able to understand the AI system's output
- **Traceability**: The system must log operations that affect its behavior
- **Human oversight**: Humans must be able to understand the system's state and intervene

**Merkraum's approach:**

| Requirement | Merkraum Capability |
|------------|-------------------|
| Logging of AI behavior changes | Every belief state change logged with provenance (source, cycle, timestamp) |
| Explainability of knowledge state | `list_beliefs` and `traverse_graph` expose what the agent knows and why |
| Traceability of decisions | SUPPORTS/CONTRADICTS/SUPERSEDES edges create an inspectable decision chain |
| Human oversight | All operations available via MCP tools — a human can query, inspect, and override any belief |

**What Merkraum does NOT provide:** Full AI Act conformity assessment, risk classification, or technical documentation generation. Merkraum covers the **knowledge layer** — tracing what an agent knows. System-level compliance requires additional components.

### Product Liability Directive (PLD 2024/2853)

The revised PLD explicitly includes AI and software under strict product liability for the first time since 1989. Key implications:

- If an AI system causes damage, the **burden of proof favors the plaintiff** (the organization deploying the AI must demonstrate the system was not defective)
- **Defect** includes: incorrect information, unsafe behavior, failure to perform as expected

**Merkraum's approach:**

When an AI agent's decision is questioned, Merkraum provides:

1. **Belief provenance**: When was this knowledge created? From what source?
2. **Contradiction history**: Were there conflicting signals? Were they surfaced?
3. **Supersession trail**: What did the agent believe before? What replaced it and why?
4. **Confidence state**: Was the agent uncertain about this claim? Was uncertainty flagged?

This creates an **evidence trail** that demonstrates the system's knowledge state at any point in time — relevant for establishing whether a defective output was caused by knowledge failure.

### GDPR — Right to Deletion and Data Minimization

**Article 17 (Right to Erasure):**

Merkraum's fixed schema means entities related to a specific data subject can be identified and removed:

```
# Find all nodes related to a person
traverse_graph(entity="Jane Doe")

# Delete specific nodes (via admin API)
# Cascading: related beliefs, edges, and vector embeddings are removed
```

**Data Minimization (Article 5):**

- Fixed schema (10 node types) constrains what is stored — the system cannot accumulate arbitrary personal data categories
- Graph dreaming consolidation reduces episodic detail to abstract patterns, naturally minimizing retained personal information
- TTL-based invalidation (for news/events) removes time-sensitive data automatically

**Cross-border Data Transfer:**

Merkraum's self-hosted option runs entirely on-premises — no data leaves the deployment environment. The hosted version runs in AWS Frankfurt (eu-central-1). No US data transfer. No CLOUD Act exposure.

## Audit Scenarios

### Scenario 1: Compliance officer audit

*"What does our AI believe about supplier X's reliability?"*

```
1. search_knowledge("supplier X reliability")
   → Returns all knowledge nodes mentioning supplier X

2. list_beliefs(status="active")
   → Filter to beliefs about supplier X

3. traverse_graph("Supplier X is a reliable partner")
   → Shows: SUPPORTS (3 positive reviews), CONTRADICTS (1 late delivery report)
   → Shows: created at Z450, last updated Z980
   → Shows: confidence 0.6 (uncertain — contradicting evidence exists)
```

### Scenario 2: Post-incident investigation

*"Why did the agent recommend vendor Y when vendor Z was cheaper?"*

```
1. traverse_graph("Recommend vendor Y for cloud migration")
   → SUPPORTS: "Vendor Y has ISO 27001 certification" (conf 0.9)
   → SUPPORTS: "Vendor Y completed similar migration for client A" (conf 0.8)
   → SUPERSEDES: "Vendor Z is the cheapest option" (superseded because:)
      → CONTRADICTS: "Vendor Z failed security audit Q2" (conf 0.85)

2. Full provenance chain shows:
   - Initial cost comparison (Z100): Vendor Z recommended
   - Security audit results ingested (Z150): contradiction detected
   - Recommendation updated (Z151): Vendor Y preferred
   - All transitions logged with timestamps and sources
```

### Scenario 3: GDPR deletion request

*"Customer has requested deletion of all their data."*

```
1. traverse_graph("Customer Name")
   → All connected nodes (beliefs, projects, events, artifacts)

2. Delete all nodes and relationships in the customer's subgraph
   → Cascading removal from Neo4j + Qdrant

3. Verify: search_knowledge("Customer Name") returns empty
   → Deletion confirmed
```

## Limitations

Merkraum is a **knowledge layer**, not a complete compliance solution:

- Does not classify AI systems by risk level (AI Act Art. 6)
- Does not generate technical documentation (AI Act Art. 11)
- Does not provide conformity assessment (AI Act Art. 43)
- Does not prevent the AI from making incorrect decisions — it makes the knowledge basis inspectable
- Contradiction detection depends on LLM quality — false negatives are possible
- GDPR deletion removes from the Merkraum graph but cannot guarantee removal from upstream LLM training data

## EU Data Residency

| Component | Location | Provider |
|----------|----------|----------|
| Neo4j Aura | Frankfurt (eu-central-1) | Neo4j Inc. |
| Pinecone (hosted) | Ireland (eu-west-1) | Pinecone |
| Qdrant (self-hosted) | Customer choice | Self-deployed |
| Application server | Frankfurt (eu-central-1) or self-hosted | AWS / self |

Self-hosted deployment: all components run locally via Docker Compose. Zero external dependencies.

---

*See also: [Architecture](architecture.md) for technical design, [Belief Tracking](belief-tracking.md) for developer guide.*
