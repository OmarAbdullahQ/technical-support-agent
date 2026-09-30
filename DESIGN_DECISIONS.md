# Design Decisions

## 1. System Architecture

The project uses a multi-model architecture instead of relying on one model for every support task.

- Model A: intent classification
- Model B: extractive question answering
- Model C: troubleshooting and response generation
- LangGraph: routing and orchestration
- Tools: retrieval, diagnostics, tickets, logs, escalation, and utility operations
- Langfuse: observability and tracing
- FastAPI: OpenAI-compatible API
- Open WebUI: user interface

Only one public model is exposed to clients:

`tuwaiq-tech-support-agent`

The internal specialist models and routing decisions remain hidden from the client.

---

## 2. Intent Taxonomy

Model A uses the following six labels:

- `api`
- `billing`
- `cancellation`
- `complaint`
- `technical`
- `upgrade`

These labels were selected from the actual training dataset rather than using the example taxonomy from the project guide.

The classes represent different support request categories and are used as an intermediate signal for routing.

Intent-to-route mapping:

| Intent | Route |
|---|---|
| api | tools |
| technical | support |
| billing | support |
| cancellation | support |
| complaint | support |
| upgrade | support |

---

## 3. Classifier Confidence Policy

The classifier confidence threshold is:

`<CLASSIFIER_THRESHOLD>`

Routing follows this policy:

1. Apply hard deterministic rules.
2. If no hard rule matches, run the fine-tuned intent classifier.
3. If classifier confidence is greater than or equal to the configured threshold, use the classifier route.
4. If confidence is below the threshold, use the LLM router as a fallback.

This prevents uncertain classifier predictions from being treated as reliable routing decisions.

---

## 4. Hard Rules

Hard rules run before the classifier.

They are used for cases where deterministic behavior is safer or more reliable.

Examples include:

- trusted documentation questions → `qa`
- health, logs, tickets, diagnostics, and live-system requests → `tools`
- data corruption → `escalate`
- security incidents → `escalate`
- production outages → `escalate`

Safety-related escalation rules are intentionally evaluated before model-based routing.

---

## 5. When QA Wins Over Generation

The QA route is used when the user explicitly asks for information that should come from trusted documentation.

Examples include:

- "According to the deployment guide..."
- "What does the documentation say..."
- questions referring to the manual or KB

The flow is:

User question → Knowledge-base retrieval → Model B extractive QA

Free generation is not preferred for these requests because Model B is intended to extract an answer from trusted context rather than generate unsupported information.

---

## 6. Knowledge Base Retrieval

The local knowledge base contains Markdown documents.

Retrieval is implemented using TF-IDF.

The retrieval flow is:

User question → TF-IDF search → relevant passage → Model B

TF-IDF was selected because it is:

- lightweight
- deterministic
- local
- inexpensive
- sufficient for the scope of the project

A limitation is that TF-IDF depends strongly on lexical overlap.

Possible future improvements include document chunking and embedding-based retrieval.

---

## 7. Router Design

Three routing strategies were implemented.

### Router A — Rules + Fine-Tuned Classifier

Uses hard rules first, followed by Model A.

### Router B — LLM Router

Uses Groq to select one of:

- `qa`
- `tools`
- `support`
- `escalate`

The LLM router uses few-shot examples and returns structured JSON.

Pydantic validation rejects malformed routes, invalid confidence values, extra fields, and invalid JSON.

### Hybrid Router

The final hybrid flow is:

Hard rules → high-confidence classifier → LLM fallback

The LLM router is only used when the classifier is uncertain.

---

## 8. Router Evaluation

The routers were compared using 8 manually labeled routing cases.

| Router | Accuracy | Macro F1 | Mean Latency |
|---|---:|---:|---:|
| Rules + Classifier | 1.000 | 1.000 | 0.0058 s |
| LLM Router | 0.625 | 0.525 | 0.7200 s |
| Hybrid | 0.875 | 0.8667 | 0.1396 s |

The LLM router achieved a JSON-valid rate of 1.00.

Observed LLM routing errors included:

- PostgreSQL connection-pool issue → `support` instead of `tools`
- API 503 with degraded database → `escalate` instead of `tools`
- QLoRA explanation → `qa` instead of `support`

The hybrid router used the LLM in 25% of the evaluated cases.

Router A performed best on this small evaluation set. The hybrid architecture is still retained to provide a fallback path for low-confidence and ambiguous inputs.

The 8-case set is an integration evaluation and should not be interpreted as a production-scale benchmark.

---

## 9. Model A Decision

Model:

`distilbert-base-uncased`

Task:

Sequence classification

Final test performance:

- Accuracy: 0.9933
- Macro F1: 0.9933

Model A is used both for intent prediction and as an input to routing decisions.

---

## 10. Model B Decision

Model:

`distilbert-base-uncased`

Task:

Extractive question answering

Held-out evaluation:

- Exact Match: 0.124
- Token F1: 0.444

The intended QA quality gate was not met.

Model B is retained in the architecture because the project requires an extractive QA specialist, but its current performance is documented as a limitation rather than treated as a successful quality gate.

---

## 11. Model C Decision

Base model:

`HuggingFaceTB/SmolLM2-135M-Instruct`

Training method:

PEFT / LoRA

LoRA configuration:

- Rank (`r`): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: `q_proj`, `v_proj`

QLoRA is used when the appropriate CUDA/bitsandbytes environment is available.

Evaluation:

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Loss | 3.219 | 2.806 |
| Perplexity | 25.00 | 16.54 |
| ROUGE-L | 0.0463 | 0.0561 |

Although loss, perplexity, and ROUGE-L improved, the fine-tuned model regressed on the Golden Set.

Therefore Model C did not pass the complete behavioral quality gate.

This limitation is explicitly documented.

---

## 12. Quality Gate Policy

A model is not considered successful only because training completed.

The project evaluates models using task-specific metrics:

- Model A: Accuracy, Precision, Recall, Macro F1
- Model B: Exact Match and token F1
- Model C: loss, perplexity, ROUGE-L, and Golden Set behavior
- Router: routing accuracy, Macro F1, latency, fallback behavior, and error analysis

A regression on required Golden Set behaviors is treated as a failure even when average metrics improve.

---

## 13. Tool Policy

All 13 required tool contracts are exposed:

1. `knowledge_base_search`
2. `ticket_search`
3. `ticket_create`
4. `system_health_check`
5. `log_analyzer`
6. `documentation_search`
7. `package_lookup`
8. `sql_query`
9. `calculator`
10. `file_search`
11. `web_search`
12. `escalate_to_human`
13. `diagnostic_runbook`

Core tools are implemented locally where practical.

Some external-system tools use deterministic mocks because the project focuses on orchestration rather than integrating every external backend.

Mock results are explicitly identified as mock data.

Read-only operations are preferred automatically.

High-risk or unsupported situations are routed to human escalation.

---

## 14. Regression Testing

Regression tests were created from failures observed during integrated testing.

Examples include:

- ensuring `corrupted` database incidents trigger escalation
- preventing requests with no documentation from retrieving unrelated KB answers
- ensuring supplied diagnostic results are routed for support synthesis

These regression cases are implemented in `tests/test_routing.py`.

---

## 15. Database

SQLite is used for local ticket persistence:

`sqlite:///data/mock_support.db`

SQLite was selected instead of PostgreSQL because:

- the project is a compact technical-support prototype
- local persistence is sufficient
- setup is simpler
- no external database service is required

The database abstraction still supports a future migration to PostgreSQL.

---

## 16. LLM Provider

Groq is used internally for the LLM router.

The public API remains OpenAI-compatible.

This separates the internal model provider from the client-facing API contract.

Open WebUI therefore interacts only with the unified support-agent endpoint.

---

## 17. LangGraph

LangGraph coordinates four primary execution paths:

- QA
- tools
- support
- escalation

The router decides which path handles each request.

This keeps specialist responsibilities explicit instead of allowing one generative model to handle every task.

---

## 18. Escalation Policy

High-risk cases bypass normal troubleshooting when deterministic escalation is safer.

Examples include:

- suspected data corruption
- security incidents
- production outages
- unresolved critical failures

Escalation creates a support ticket and transfers responsibility to a human operator.

---

## 19. Observability

Langfuse is used for workflow tracing.

Each request carries:

- `request_id`
- `trace_id`
- routing metadata
- model/tool execution information

The Langfuse callback is passed into the LangGraph invocation so agent execution can be inspected end-to-end.

Local logs also record routing decisions and latency.

---

## 20. API Design

FastAPI exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

The API uses Bearer authentication.

Only one model is exposed:

`tuwaiq-tech-support-agent`

The response includes internal metadata such as:

- route
- intent
- source
- trace ID
- end-to-end latency
- router latency
- escalation status

---

## 21. Streaming

The API supports buffered Server-Sent Events for OpenAI-compatible streaming.

This is not true token-by-token model generation.

The decision was made to provide client compatibility without adding unnecessary generation complexity.

---

## 22. Docker and Open WebUI

The application runs through Docker Compose.

The stack includes:

- support-agent
- Open WebUI

The support agent runs on port `8000`.

Open WebUI runs on port `3000` and connects to:

`http://support-agent:8000/v1`

This ensures Open WebUI communicates with the unified agent rather than directly with specialist models.

---

## 23. Deployment Decision

The application is fully containerized and runs locally using Docker Compose.

Dokploy deployment was not completed in the current project iteration because no remote hosting environment was provisioned.

The current deliverable therefore demonstrates local containerized deployment rather than a public HTTPS deployment.

