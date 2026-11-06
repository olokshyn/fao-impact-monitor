**Athena — updated system-design draft**

Athena is a deployable research system built around a persistent evidence graph and **incremental, retrieval-driven relation discovery**. LLM agents extract claims, propose connections, and extend the relation hierarchy. A separate verification stage controls what becomes usable knowledge.

Athena maintains one reusable, continuously evolving KB. Discovery happens during ingestion, through targeted background processing, and when research questions expose missing connections.

**System composition**

| Component                | Responsibility                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| FastAPI service          | Accept documents and claims, start research jobs, expose filtered graph queries, and return results and citations. |
| Ingestion agent          | Extract atomic claims, resolve entities, normalize quantities, and preserve source passages and tags.              |
| Candidate retriever      | Find potentially connected claims through semantic, lexical, entity, and graph indexes.                            |
| Relation-discovery agent | Propose relations, their direction, applicability conditions, and supporting evidence.                             |
| Ontology agent           | Reuse existing relation definitions or propose specific subtypes.                                                  |
| Verification agent       | Independently check claims and relations against original evidence and applicable inference rules.                 |
| Research agent           | Decompose questions, explore evidence chains, investigate gaps, and synthesize answers.                            |
| Background workers       | Run ingestion, discovery, verification, targeted backfills, index synchronization, and maintenance.                |
| Storage layer            | Jena for authoritative knowledge; filesystem/GCS for files; MongoDB for a rebuildable retrieval index.             |

These are logical responsibilities; they can share workers and model infrastructure.

**Knowledge and evidence model**

Store the following connected records in Jena:

| Record              | Required information                                                                                                                               |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source              | Original identifier/URL/DOI, publication metadata, source tags, stored snapshot, content hash, and version.                                        |
| Evidence            | Exact passage or data excerpt; source version; page, section, table/cell, or text offsets.                                                         |
| Claim               | Original wording, normalized proposition, entities/processes, quantities, units, location, time, population, conditions, tags, and evidence links. |
| Relation assertion  | Subject, object, relation type, direction, conditions, evidence links, derivation if applicable, and verification status.                          |
| Relation definition | Meaning, parent types, permitted endpoints, directionality, examples, and version.                                                                 |
| Verification record | Checks performed, result, justification, verifier/model version, and timestamp.                                                                    |

Causality connects the **phenomena described by claims**; support and contradiction connect propositions. Canonical process and variable nodes connect claims about the same phenomenon without merging their evidence or contexts.

Every accepted claim and relation must have traceable original evidence. A URL alone is insufficient: citations must resolve to the particular passage or data supporting the assertion.

Keep scientific status separate from verification status:

- **Source-reported:** the source explicitly states the claim or relation.
- **Derived:** Athena provides cited premises, a recorded derivation, and explicit assumptions.
- **Hypothetical:** a research candidate awaiting adequate evidence; excluded from the accepted knowledge graph.

“Verified” means the evidence supports the recorded statement and its qualifications. It does not mean the statement is universally true.

**Configurable tags and filtered reasoning**

Define a versioned tag registry specifying each field’s name, value type, vocabulary, multiplicity, extraction rules, and inheritance rules. Initially include:

- Country.
- Region within a country.
- Broader geographical regions, including overlapping classifications.
- Source type: FAO publication, peer-reviewed article, web page, etc.
- Publishing group, publisher, and journal as separate fields.
- Acquisition channel, such as web search or direct upload.

Keep acquisition channel separate from publication type: a peer-reviewed article found through web search remains peer-reviewed. Users can filter either dimension.

Preserve supplied metadata and extract missing tags when supported. Unknown values remain unknown. Geography must distinguish **where evidence applies** from places merely mentioned in a document. Geographic ancestors may be added through an explicit location hierarchy, with that derivation recorded.

Implement filtering through indexed RDF predicates and matching metadata in the retrieval index:

- Support inclusion, exclusion, AND/OR combinations, and explicit treatment of unknown values.
- Apply filters **before candidate ranking and at every graph expansion**, not only to the final answer.
- In strict mode, every participating claim and relation assertion must satisfy the filters. Shared concept/schema nodes provide structure, not additional evidence.
- Enable a relation only when it has an admissible supporting evidence set. Every required premise of a derived relation must also pass.
- Do not let excluded evidence influence confidence, conflict summaries, or cached answers indirectly.

For example, disabling web-page sources removes their support even if their claims were merged with claims from a journal article. Independently sufficient journal evidence may still support the assertion.

A Kenya-only filter excludes evidence without Kenya applicability. An optional, explicit transferability mode may include broader evidence, clearly labeled as extrapolation. It must never silently relax the filter.

Adding a new indexed tag field may require an index rebuild and targeted metadata backfill; ordinary ingestion does not.

**Relation hierarchy and contradictions**

Start with:

- `causes → increases / decreases`
- `correlatesWith → positivelyCorrelatesWith / negativelyCorrelatesWith`
- `supports`
- `contradicts`

“Increases/decreases” describes effect direction, not benefit or harm. Domain-specific types extend these families, for example:

`yieldLoss → decreases → causes`

Define `yieldLoss` precisely, including the agricultural variable being reduced.

Use `rdfs:subPropertyOf` for predicate inheritance. Parent queries include descendants. Because relations are stored as assertion records with metadata, queries explicitly join the recorded relation type to its ancestors. Incoming/outgoing causal queries select the corresponding endpoint; correlation and contradiction queries account for symmetry, while support remains directional. [Jena inference](https://jena.apache.org/documentation/inference/), [SPARQL property paths](https://jena.apache.org/documentation/query/property_paths.html)

Athena preserves contradictory claims, relation assertions, and supporting evidence. Add specific contradiction types when justified, such as `oppositeEffectDirection` or `incompatibleEstimate`, under `contradicts`.

A contradiction assertion cites both sides and explains their incompatibility. Differences in population, exposure, period, or uncertainty may explain disagreement; they must not automatically become contradictions. Conflicting assertions remain individually retrievable, with their evidence and contexts intact.

**Verification before acceptance**

The verification stage checks:

1. **Citation integrity:** the cited passage/data exists in the stored source version.
2. **Faithful extraction:** wording, quantities, units, negation, uncertainty, and context match the evidence.
3. **Relation support:** evidence supports the specific relation, direction, and subtype.
4. **Context compatibility:** linked claims have compatible variables, populations, geography, timing, and conditions.
5. **Derivation validity:** derived assertions identify every premise, rule/model, and additional assumption.
6. **Schema validity:** endpoints, tags, required provenance, and ontology constraints are valid.

For a source-reported `causes` relation, the cited original claim must actually express causality. Correlation, co-occurrence, and semantic similarity are insufficient. Preserve whether the source’s wording is tentative and what study design supports it.

A multi-hop chain may support a qualified derived mechanism, but cannot be presented as a directly reported causal finding. Plausible unsupported links remain research candidates.

Use deterministic citation/schema checks plus a separate evidence-reading verifier. Ambiguous cases remain pending or enter review. LLM verification alone cannot guarantee correctness.

**Incremental discovery lifecycle**

1. **Ingest and normalize.**  
   Store the source snapshot, extracted evidence, claims, and tags. Verify claims and any explicitly stated relations. Update only affected index entries and concept profiles.

2. **Retrieve a bounded candidate set.**  
   Combine semantic similarity, lexical matching, shared entities, graph neighbors, and complementary mechanisms. For example, match an effect on soil organisms with claims about those organisms’ role in food availability.

   Athena’s memory consists of indexed claims and compact concept/type profiles containing aliases, representative claims, and known mechanisms. The LLM retrieves relevant memory rather than reading the entire KB.

3. **Discover relations.**  
   Evaluate both directions against candidates, respecting context. Propose evidence-backed connections and submit them to verification.

4. **Extend the ontology when necessary.**  
   Check existing definitions and synonyms first. A new type needs a definition, parent, endpoint constraints, and cited examples. Validate the hierarchy before activation. Creating a type does not establish any particular relation.

5. **Backfill selectively.**  
   Each activated type gets a retrieval recipe based on endpoint types, aliases, and examples. Background workers evaluate relevant older claims, including old-to-old pairs, in checkpointed batches. Adding a relation subtype does not require re-embedding the corpus.

6. **Discover during research.**  
   When a question exposes a missing link, search admissible stored claims and original passages. Run new claims, relations, and types through the same verification pipeline before persisting them. External retrieval, when enabled, follows normal ingestion and tagging.

7. **Maintain continuously.**  
   Revisit unresolved claims and underconnected areas. Deduplicate assertions, maintain type aliases, and invalidate dependent derivations after corrections, retractions, or changed evidence. Preserve history; do not delete scientific disagreement as cleanup.

Bounded retrieval avoids comparing every new claim with every existing claim. It trades exhaustive discovery for manageable cost; background exploration and research-driven discovery improve recall over time.

**Storage decisions**

**Jena TDB2 + Fuseki is the authoritative knowledge store.** It holds claims, relation assertions, evidence metadata and excerpts, tags, ontology versions, verification records, and derivation dependencies. Separate named graphs can distinguish accepted knowledge, candidates, ontology, and operational metadata. TDB2 provides persistent RDF storage and integrates with Fuseki. [TDB2 documentation](https://jena.apache.org/documentation/tdb2/)

**MongoDB is not necessary as a second raw-evidence database.** Extracted text can be represented in Jena, while large extraction outputs are better retained as files. Duplicating authoritative evidence across two databases would add synchronization work.

The proposed deployment uses MongoDB specifically for **rebuildable semantic/lexical retrieval records**: stable Jena IDs, searchable text, embeddings, filter metadata, and record versions. MongoDB Vector Search supports indexed pre-filter fields. Use a deployment with Search/Vector Search enabled; a plain MongoDB server alone is insufficient. [Index documentation](https://www.mongodb.com/docs/vector-search/indexes/vector-search-type/), [deployment documentation](https://www.mongodb.com/docs/search/self-managed/current/)

Publish committed Jena changes to the index through a persistent update queue. Recheck candidate IDs, versions, tags, and verification status in Jena before use, so stale index entries cannot bypass restrictions.

**Binary and large-file storage uses interchangeable adapters:** local filesystem or Google Cloud Storage, selected by configuration. Both expose the same read/write/metadata interface and use stable logical object IDs.

Store original PDFs, images, source snapshots, OCR results, and large extraction files there. Citations reference immutable content hashes/versions, not machine-specific paths. On GCS, retain the object generation identifier. Switching backends copies and verifies objects without changing claim IDs or citations. [GCS object version identifiers](https://docs.cloud.google.com/storage/docs/objects)

**Deployment and API**

Deploy containerized FastAPI, workers, a scheduler, Fuseki/TDB2, MongoDB Search, and a durable job broker. Celery with RabbitMQ is a reasonable initial worker setup. Heavy ingestion and research run outside FastAPI request processes. [FastAPI background-work guidance](https://fastapi.tiangolo.com/tutorial/background-tasks/)

Minimum endpoints:

| Endpoint                     | Purpose                                                             |
| ---------------------------- | ------------------------------------------------------------------- |
| `POST /ingestions`           | Submit documents, source references, or cited raw claims with tags. |
| `POST /research`             | Submit a question, filters, and research limits.                    |
| `GET /jobs/{id}`             | Return progress, failures, and completed results.                   |
| `GET /claims/{id}/relations` | Query incoming/outgoing relations by parent type and filters.       |
| `GET /evidence/{id}`         | Retrieve the exact cited evidence and source location.              |

Long-running requests return `202 Accepted` with a job ID. Jobs use retries, idempotent writes, checkpoints, and persistent status. A committed job/event record allows recovery if broker publication fails.

Batch graph updates through short transactions; parallelize extraction and verification outside them. Keep persistent storage, backups, health checks, and queue monitoring in the deployment configuration. Benchmark graph size and write throughput before committing to corpus-scale capacity.

**Research reasoning and example questions**

The research agent decomposes a question into variables, contexts, and subquestions, then explores paths in bounded rounds. Each step retains evidence, applicability conditions, and competing explanations.

**Paths identify candidate mechanisms; they do not automatically establish causal effects.** Neither causality nor correlation should be treated as generally transitive. Quantitative predictions require explicit models, assumptions, and uncertainty estimates; chain confidence must not be obtained by blindly multiplying LLM scores.

**Example 1 — microplastics and biodiversity in Kenya**

> How does microplastic exposure in Kenyan agricultural soils affect soil biodiversity and associated food webs? Distinguish polymer type, particle size/shape, concentration, exposure duration, soil properties, and land management. Assess effects on microbial community composition, soil-invertebrate abundance and diversity, plant productivity, and food availability for dependent species. Which effects have direct Kenyan field evidence, and which rely on laboratory studies or evidence transferred from other regions?

Candidate pathways to investigate include:

`microplastic exposure → soil physical/chemical changes → soil-community changes → nutrient cycling or plant productivity → food availability → population effects`

Also examine direct toxicity and alternative pathways. Do not assume every effect is negative or every link established. Experimental research explicitly distinguishes particle properties and concentrations, supporting this more precise question formulation. [Example experimental study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7920964/)

Return separate Kenyan-only and explicitly broadened evidence views, with effect estimates where supported, contradictions, and missing links.

**Example 2 — El Niño and Kenya’s economy**

> Using forecasts available as of a specified date, how could the next forecast El Niño episode affect Kenya’s real GDP and agricultural value added over the following 12–24 months, relative to an explicit ENSO-neutral baseline? Distinguish counties, rainfall seasons, crop and livestock systems, irrigation coverage, and the Indian Ocean Dipole. Evaluate beneficial rainfall and adverse flooding, waterlogging, or locally forecast rainfall deficits, together with effects on infrastructure, trade, prices, and household income.

Candidate pathways include:

`ENSO/IOD conditions → regional seasonal weather probabilities → crop/livestock and infrastructure impacts → production, transport, trade and prices → sectoral value added → GDP`

Kenya should not be assigned a universal “El Niño causes drought” relation: regional outlooks can indicate enhanced October–December rainfall, requiring location- and season-specific reasoning. [WMO regional outlook](https://public.wmo.int/media/news/el-nino-impacts-greater-horn-of-africa)

Verify irrigation statistics’ year and denominator. Unirrigated land is not equivalent to lost production, export changes are not automatically proportional to GDP changes, and aggregation must avoid double-counting.

Return conditional scenarios with uncertainty ranges where a defensible quantitative model exists; otherwise return a qualitative assessment and the missing data/model requirements.

**Acceptance criteria**

The prototype must demonstrate accurate citations, evidence-supported relations, descendant-type retrieval, preservation of contradictions, strict filtering throughout derivations, targeted backfill, discovery during retrieval, restart-safe jobs, interchangeable file storage, and measurable relation precision and candidate-retrieval recall. Answers must expose evidence, assumptions, and unresolved gaps.

# Implementation details

1. Athena must be implemented in python 3.13 following the latest python 3.13 coding conventions.
   1. Type hints are mandatory
2. Background worker must implement batching for efficiency: only run once a given number of tasks of a certain type has accumulated or once a fixed time has passed.
3. FastAPI can accept documents and claims in batches
4. Use OpenAI Agents SDK for implementing the agents
5. Every decision made by an agent and every processed claim must be logged for full observability
