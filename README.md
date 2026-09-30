# Multi-Model Agentic Technical Support System

A production-style technical support agent built around multiple specialized models, hybrid routing, domain tools, LangGraph orchestration, Langfuse observability, an OpenAI-compatible FastAPI API, Docker, and Open WebUI.

The system exposes one public model:

`tuwaiq-tech-support-agent`

while internally selecting the appropriate specialist model or tool.

---

## Architecture

```mermaid
flowchart TD
    U[User / Open WebUI] --> API[FastAPI OpenAI-Compatible API]

    API --> R[Hybrid Router]

    R -->|Documentation| QA[QA Route]
    R -->|Diagnostics / Live Data| T[Tools Route]
    R -->|General Support| S[Support Route]
    R -->|High Risk| E[Escalation Route]

    QA --> KB[TF-IDF Knowledge Base Search]
    KB --> MB[Model B: Extractive QA]

    T --> TOOLS[Support Tools]
    TOOLS --> MC[Model C: Support Specialist]

    S --> MC

    E --> HUMAN[Human Escalation / Ticket]

    R -. Low confidence .-> LLM[Groq LLM Router]

    API -. Traces .-> LF[Langfuse]