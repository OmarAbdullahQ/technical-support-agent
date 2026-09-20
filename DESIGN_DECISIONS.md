# Design Decisions

## 1. Architecture

The system uses a multi-model architecture instead of relying on one model for all tasks.

- Model A: intent classification
- Model B: extractive QA
- Model C: troubleshooting and response generation

LangGraph coordinates routing, tools, specialist models, and escalation.

The public API exposes only one model:

`tuwaiq-tech-support-agent`

Internal model selection is hidden from Open WebUI and API clients.

---

## 2. Intent Taxonomy

The final intent labels are:

- `api`
- `billing`
- `cancellation`
- `complaint`
- `technical`
- `upgrade`

These labels were selected from the actual training dataset instead of using the example taxonomy from the project guide.

The classifier returns both the predicted intent and confidence score.

---

## 3. Routing Policy

The routing system uses three layers:

1. Hard rules
2. Fine-tuned intent classifier
3. LLM fallback for low-confidence cases

Hard rules are evaluated first for cases where deterministic routing is safer or clearer, including:

- documentation questions
- tool and diagnostic requests
- security incidents
- data corruption
- production outages

The classifier intent-to-route mapping is:

- `api` → `tools`
- `technical` → `support`
- `billing` → `support`
- `cancellation` → `support`
- `complaint` → `support`
- `upgrade` → `support`

When classifier confidence is below the configured threshold, the request is sent to the LLM router.

---

## 4. Router Comparison

Three routing approaches were evaluated using 8 labeled routing cases.

| Router | Accuracy | Macro F1 | Mean Latency |
|---|---:|---:|---:|
| Rules + Classifier | 1.00 | 1.00 | 0.0058 s |
| LLM Router | 0.625 | 0.525 | 0.7200 s |
| Hybrid | 0.875 | 0.8667 | 0.1396 s |

The LLM router achieved a JSON-valid rate of 1.00.

The main observed LLM routing errors were:

- database pool issue routed to `support` instead of `tools`
- API 503 issue routed to `escalate` instead of `tools`
- QLoRA question routed to `qa` instead of `support`

The hybrid router used the LLM fallback in 25% of the evaluation cases.

The rules + classifier router performed best on this small evaluation set. However, the hybrid policy is retained because the LLM fallback provides a recovery path for ambiguous or low-confidence classifier predictions.

The evaluation set is small and should be treated as an integration comparison, not a production benchmark.

---

## 5. Knowledge Base Retrieval

The knowledge base uses local Markdown documents.

Retrieval is implemented using TF-IDF.

The flow is:

User question → TF-IDF retrieval → relevant context → Model B QA

TF-IDF was selected because it is:

- lightweight
- deterministic
- local
- easy to evaluate
- sufficient for the project scope

A known limitation is that retrieval depends mainly on lexical overlap.

Future improvements could include:

- document chunking
- embedding-based retrieval
- retrieval evaluation using MRR and Recall@K

---

## 6. Specialist Model Decisions

### Model A

`distilbert-base-uncased`

Used for intent classification and routing confidence.

### Model B

`distilbert-base-uncased`

Used for extractive QA over retrieved technical context.

Model B did not fully meet the intended quality gate, so this limitation is documented.

### Model C

`HuggingFaceTB/SmolLM2-135M-Instruct` with a PEFT adapter.

Used for troubleshooting synthesis and general support responses.

Model C showed improvement in some evaluation metrics but did not pass all required behavioral checks. It is retained as part of the project architecture while this limitation is documented.

---

## 7. Tool Layer

All required tool contracts are exposed.

The system includes:

- knowledge base search
- ticket search
- ticket creation
- system health check
- log analysis
- documentation search
- package lookup
- SQL query
- calculator
- file search
- web search
- escalation
- diagnostic runbook

Some tools are fully functional while others use deterministic mock implementations.

This follows the project requirement that all tool contracts exist while only a subset must be fully implemented.

Mock outputs are explicitly marked as mock data.

---

## 8. Database

SQLite is used for ticket persistence:

`sqlite:///data/mock_support.db`

SQLite was selected instead of PostgreSQL because the project only requires lightweight local persistence.

The implementation still keeps database access isolated enough to allow a future migration to PostgreSQL.

---

## 9. LLM Provider

Groq is used internally for the LLM router.

The public API remains OpenAI-compatible.

This separates:

- internal model provider
- external API contract

Open WebUI therefore communicates with the application without needing to know which provider or internal model is used.

---

## 10. Escalation Policy

High-risk cases are routed directly to escalation when possible.

Examples include:

- suspected data corruption
- security incidents
- production outages
- unresolved critical failures

These cases bypass normal generation because deterministic escalation is safer than relying on a generated troubleshooting response.

---

## 11. Observability

Langfuse is used for tracing the agent workflow.

Each request includes:

- `request_id`
- `trace_id`
- routing metadata
- model/tool execution metadata

The Langfuse callback is passed into the LangGraph invocation so routing and execution can be traced end-to-end.

---

## 12. API Design

FastAPI exposes:

- `/health`
- `/v1/models`
- `/v1/chat/completions`

Authentication uses a Bearer API key.

Only one public model is exposed:

`tuwaiq-tech-support-agent`

This keeps the internal multi-model architecture hidden from the client.

---

## 13. Streaming

The API supports buffered SSE streaming.

This provides compatibility with Open WebUI streaming behavior, but it is not true token-by-token model generation.

---

## 14. Deployment

The application is containerized using Docker.

Docker Compose runs:

- support-agent
- Open WebUI

The deployment design supports Dokploy deployment and HTTPS exposure.

Model files and data are mounted into the support-agent container.

---

