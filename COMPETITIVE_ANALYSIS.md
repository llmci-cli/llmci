# Scaffold — Competitive Analysis

Last updated: May 2026

---

## Executive Summary

The LLM evaluation space has matured rapidly, with tools roughly splitting into two camps: **open-source CLI frameworks** for developer-led testing and **commercial platforms** for team-wide observability and collaboration. Most tools focus on either prompt exploration (dev-time) or production monitoring (runtime). Very few are designed specifically as **CI-native quality gates** — tools that live in the PR workflow and block regressions before merge.

Scaffold's positioning occupies a gap: it is a safety gate, not a dashboard. Its closest competitor is Promptfoo, but Promptfoo's March 2026 acquisition by OpenAI raises questions about provider neutrality. Scaffold's unique advantages are automated model migration, agentic workflow testing, and eval dataset generation — none of which exist in the current competitive set.

---

## Competitive Landscape Map

| | Primary Use | Interface | CI Gate | Migration | Agent Eval | Dataset Gen | License |
|---|---|---|---|---|---|---|---|
| **Scaffold** | PR quality gate | CLI + YAML | Native | Automated | Full (trajectory + constraints) | Yes | Open source |
| **Promptfoo** | Prompt comparison + red teaming | CLI + YAML | Supported | No | No | No | MIT (now OpenAI) |
| **Braintrust** | Eval platform + observability | Platform + SDK | GitHub Action | No | Tracing only | No | Closed |
| **Langfuse** | Observability + tracing | Platform (self-host or cloud) | Limited | No | Tracing only | No | MIT |
| **DeepEval** | Automated eval CI | CLI + Python | Native | No | No | Synthetic generation | Apache 2.0 |
| **DSPy** | Prompt optimization framework | Python framework | No | Re-optimize | No | No | MIT |
| **Arize Phoenix** | Eval + production monitoring | Platform + SDK | No | No | Trace support | No | Apache 2.0 |
| **RAGAS** | RAG evaluation | Python library | No | No | No | No | Apache 2.0 |

---

## Detailed Competitor Profiles

### 1. Promptfoo

**What it is:** An open-source CLI-first tool for evaluating and red-teaming LLM applications. YAML-based test definitions, support for 60+ providers, and 500+ adversarial attack vectors for security testing.

**Acquired by OpenAI, March 9, 2026.** This is the single most important competitive development. Promptfoo was the closest tool to Scaffold's positioning — CLI-native, YAML configs, CI integration. The acquisition raises significant concerns:

- **Provider neutrality.** Will Promptfoo remain unbiased across providers, or will it subtly favor OpenAI models? Enterprise teams using Anthropic, Google, or open-source models may not trust an OpenAI-owned eval tool.
- **Open-source trajectory.** Promptfoo is MIT-licensed today, but OpenAI could restrict features to a commercial offering over time.
- **Strategic direction.** OpenAI may pivot Promptfoo toward security/red-teaming (their stated focus) and away from general-purpose eval CI gating.

**Strengths:**
- Mature CLI with extensive provider support
- Strong red-teaming/security testing (142 plugins, OWASP coverage)
- GitHub Actions integration with before/after PR comparison
- YAML config is developer-friendly
- Large community (300k+ users claimed)

**Weaknesses vs. Scaffold:**
- No automated model migration
- No agentic workflow evaluation (no trace analysis, no trajectory judging)
- No eval dataset generation
- Primarily prompt-level testing — doesn't catch upstream pipeline changes
- CI integration exists but is secondary to dev-time exploration (the tool was designed for comparison, not gating)
- Post-acquisition uncertainty on neutrality

**Scaffold opportunity:** Promptfoo users concerned about the OpenAI acquisition are actively looking for alternatives. Scaffold can position as the provider-neutral, CI-native alternative with capabilities Promptfoo never had (migration, agents, dataset generation).

---

### 2. Braintrust

**What it is:** A full-stack AI observability and evaluation platform. Combines production tracing, offline evaluation, experiment tracking, and CI integration into a single cloud platform. Raised $80M Series B.

**Strengths:**
- Polished UI with side-by-side experiment comparison
- Strong diff UX for comparing eval runs
- Native GitHub Actions integration (`eval-action`) that posts results to PRs
- "Trace to dataset" — convert production traces into eval datasets with one click
- Custom database (Brainstore) optimized for AI trace data
- Enterprise features: SOC 2, HIPAA, SSO, RBAC
- Loop Agent — AI that helps improve prompts automatically
- SDKs for Python, TypeScript, Go, Ruby, C#

**Weaknesses vs. Scaffold:**
- It's a platform, not a CLI. Requires account creation, SDK integration, data leaving your infrastructure.
- No automated model migration
- No agentic trajectory evaluation (traces agents but doesn't judge trajectory quality)
- No constraint enforcement (tool call budgets, token budgets)
- Closed source — no self-hosting of the full platform
- Pricing: Free tier is limited; Pro is $249/month; Enterprise is custom
- Heavier integration burden — requires SDK instrumentation in your code

**Scaffold opportunity:** Teams that want the CI gating benefit without adopting a full platform. Scaffold is zero-infrastructure — it's a CLI that reads a YAML file and exits 0 or 1. No account, no SDK, no data leaving your repo.

---

### 3. Langfuse

**What it is:** An open-source observability platform with a trace-first architecture. Can be self-hosted (Postgres + ClickHouse) or used as a cloud service. Strong in Europe due to GDPR-friendly self-hosting.

**Strengths:**
- Fully self-hostable — important for data residency requirements
- Open source (MIT license)
- LLM-as-judge evaluation with custom rubrics
- Datasets and experiment runners for controlled testing
- Multiple score types: numeric, categorical, boolean
- Good OpenAI/Anthropic/LangChain integrations
- Active community and fast development pace

**Weaknesses vs. Scaffold:**
- Primarily an observability tool — tracing and monitoring, not CI gating
- No native CI/CD quality gate (can be scripted but not first-class)
- No automated model migration
- No agentic trajectory evaluation (traces agents but doesn't evaluate quality)
- No eval dataset generation
- Requires infrastructure to self-host (Postgres + ClickHouse)
- The eval workflow is secondary to the observability workflow

**Scaffold opportunity:** Langfuse and Scaffold are complementary, not directly competitive. Langfuse monitors production; Scaffold gates PRs. A team could use both. However, teams currently using Langfuse's eval features as a CI workaround would find Scaffold purpose-built for that workflow.

---

### 4. DSPy

**What it is:** A programming framework for LLM applications that treats prompts as typed functions and includes automated optimizers (MIPROv2, BootstrapFewShot, SIMBA, etc.) that tune prompts to maximize metrics.

**Strengths:**
- Sophisticated prompt optimization (MIPROv2 delivers 10-40% quality lift on structured tasks)
- Model-agnostic — same code runs against different providers
- Academic rigor (Stanford NLP group)
- Built-in model migration: re-optimize when switching models
- First-class agent support in DSPy 3.0

**Weaknesses vs. Scaffold:**
- Not a testing/CI tool at all — it's a development framework
- No CI integration, no PR gating, no regression detection
- Requires rewriting your prompts as DSPy programs (heavy adoption cost)
- Optimization is general-purpose, not specifically designed for migration parity
- No holdout validation or overfitting prevention in the migration flow
- No eval dataset generation
- No agentic trajectory evaluation

**Scaffold opportunity:** DSPy and Scaffold solve adjacent problems. DSPy is for building optimized prompts; Scaffold is for ensuring they don't regress. They could be used together. However, Scaffold's migration feature directly competes with DSPy's re-optimization story — and Scaffold's approach (small iterative edits with holdout validation) is more controlled and production-safe than DSPy's ground-up re-optimization.

---

### 5. DeepEval

**What it is:** An open-source framework for automated LLM evaluation in CI/CD pipelines. 14+ built-in metrics, G-Eval custom rubrics, pytest integration, and optional Confident AI SaaS for dashboard/collaboration.

**Strengths:**
- Strong CI/CD focus — pytest plugin, GitHub Actions integration
- 14+ built-in metrics covering hallucination, toxicity, bias, relevance, etc.
- Synthetic dataset generation (via Confident AI)
- Red-teaming and vulnerability scanning
- MIT license, free to use
- Python-native, integrates with existing test suites

**Weaknesses vs. Scaffold:**
- Metrics are pre-defined — less flexible for custom business-specific judges
- No automated model migration
- No agentic trajectory evaluation
- Synthetic generation is SaaS-only (Confident AI), not in the open-source CLI
- Tightly coupled to Python pytest — less flexible than command-mode black-box testing
- No baseline comparison (tests pass/fail on absolute thresholds, not relative regression)

**Scaffold opportunity:** DeepEval is the closest existing tool to Scaffold's CI philosophy. The key differentiators are migration automation, pipeline-level (not just prompt-level) testing, relative regression thresholds, and agentic evaluation. DeepEval's pytest coupling also means it only works for Python codebases; Scaffold's command mode works with any language.

---

### 6. Arize Phoenix

**What it is:** An open-source observability platform combining offline evaluation with production monitoring. OpenTelemetry-native, supports embedding drift detection, spans, and traces.

**Strengths:**
- OpenTelemetry-native design (industry-standard tracing)
- Embedding drift detection for production monitoring
- Supports offline evaluation with datasets
- Good production monitoring with real-time dashboards
- Apache 2.0 license

**Weaknesses vs. Scaffold:**
- Primarily a monitoring/observability tool
- No CI quality gate
- No automated model migration
- No agentic trajectory evaluation
- No eval dataset generation
- Requires infrastructure for the platform

**Scaffold opportunity:** Like Langfuse, Arize Phoenix is complementary. It monitors production; Scaffold gates pre-merge.

---

### 7. RAGAS

**What it is:** A purpose-built evaluation framework for RAG (Retrieval-Augmented Generation) pipelines. Evaluates retrieval quality, context relevance, faithfulness, and answer quality without requiring ground-truth data.

**Strengths:**
- Best-in-class RAG evaluation (purpose-built)
- Reference-free metrics — no gold dataset required for some evaluations
- Evaluates both retrieval and generation stages independently
- Well-adopted in the RAG community

**Weaknesses vs. Scaffold:**
- RAG-specific — doesn't generalize to classification, extraction, agents, etc.
- No CI integration
- No model migration
- No agentic evaluation
- Library, not a tool — requires custom scripting to use in CI

**Scaffold opportunity:** RAGAS could potentially be integrated as a judge type within Scaffold for RAG-specific evals, combining RAGAS's domain expertise with Scaffold's CI infrastructure.

---

## Feature Gap Analysis

### What only Scaffold does (unique differentiators)

| Feature | Description | Competitive Status |
|---|---|---|
| **Automated model migration** | Iterative prompt optimization with holdout validation, early stopping, and step-size control | No competitor offers this. DSPy has general optimization but not migration-specific with holdout validation. |
| **Pipeline-level testing** | Tests the full pipeline (retrieval + preprocessing + prompt + model), not just the prompt in isolation | All competitors test at the prompt level. Braintrust and Langfuse trace full pipelines but don't test them against gold datasets. |
| **Relative regression thresholds** | "Must not drop more than X% from main branch baseline" | Most tools only support absolute thresholds. Braintrust has some regression tracking but not configurable thresholds. |
| **Agentic trajectory evaluation** | Composite judging of outcome + trajectory efficiency + constraint enforcement | No competitor evaluates agent trajectories holistically. Braintrust/Langfuse trace agents; MLflow has some agent eval. |
| **Eval dataset generation** | Bootstrap datasets from production traces, specs, or augmentation | DeepEval/Confident AI has synthetic generation. Braintrust has "trace to dataset." Neither offers the full three-strategy approach. |
| **Zero-infrastructure CI gate** | CLI reads YAML, exits 0/1. No account, no SDK, no platform. | Braintrust requires an account. Langfuse requires infrastructure. Promptfoo is closest but was designed for exploration, not gating. |

### What competitors do that Scaffold doesn't (v1 gaps)

| Feature | Who has it | Scaffold stance |
|---|---|---|
| **Production monitoring/observability** | Langfuse, Braintrust, Arize Phoenix | Out of scope. Scaffold is pre-merge, not production. Complementary, not competitive. |
| **Red teaming / security scanning** | Promptfoo (142 plugins), DeepEval | Not planned for v1. Could be a future addition. |
| **Visual experiment comparison UI** | Braintrust (strong diff UX) | Scaffold outputs markdown reports. A dashboard is a paid-tier future feature. |
| **Embedding drift detection** | Arize Phoenix | Production monitoring concern, out of scope. |
| **RAG-specific metrics** | RAGAS, DeepEval | Could integrate RAGAS as a judge type in the future. |
| **Self-hosted platform** | Langfuse | Scaffold doesn't need hosting — it's a CLI. Self-hosting is irrelevant. |
| **Fine-tuning optimization** | DSPy (BootstrapFinetune) | Scaffold focuses on prompt tuning, not weight tuning. |

---

## Pricing Landscape

| Tool | Free Tier | Paid Tier | Model |
|---|---|---|---|
| **Scaffold** | Full CLI (open source) | Hosted service (future) | Open core |
| **Promptfoo** | Full CLI (MIT) | Cloud platform (pricing unclear post-acquisition) | Open core |
| **Braintrust** | Starter (limited) | Pro $249/mo, Enterprise custom | Freemium SaaS |
| **Langfuse** | Self-hosted (free) | Cloud from ~$59/mo | Open core |
| **DeepEval** | CLI (MIT) | Confident AI $19.99-$49.99/user/mo | Open core |
| **DSPy** | Full framework (MIT) | None | Fully open source |
| **Arize Phoenix** | Self-hosted (free) | Cloud from ~$50/mo | Open core |
| **RAGAS** | Full library (Apache 2.0) | None | Fully open source |

---

## Key Market Dynamics

### The Promptfoo acquisition shakes the market

OpenAI's acquisition of Promptfoo in March 2026 is the most significant recent event in this space. Promptfoo was the de facto standard for CLI-based LLM eval with 300k+ users. The acquisition creates:

1. **A trust gap.** Multi-provider teams (using Anthropic, Google, Mistral, open-source models) may not want their eval infrastructure owned by one provider. This is similar to how companies avoid using AWS's monitoring tools if they're multi-cloud.

2. **A migration wave.** Teams are actively evaluating Promptfoo alternatives. Multiple "Promptfoo alternatives" comparison posts have appeared since March 2026.

3. **A strategic pivot risk.** OpenAI likely acquired Promptfoo for the red-teaming/security capabilities, not the general eval features. The general eval CLI may stagnate or be deprioritized.

Scaffold should explicitly position against this: **"Provider-neutral, community-owned LLM testing. Not owned by any model provider."**

### The eval-to-observability pipeline is consolidating

Braintrust's "trace to dataset" and Langfuse's experiment runners show a trend: platforms want to own the full loop from production monitoring → dataset creation → evaluation → CI gating. Scaffold's approach is different — it's a focused, composable tool that does one thing well (CI gating) and leaves the rest to purpose-built tools.

This is a deliberate positioning choice. The "do one thing well" Unix philosophy resonates with engineers who don't want to adopt a full platform for a CI check.

### Agentic evaluation is wide open

Despite the explosion of agent frameworks (OpenAI Agent SDK, Claude Agent SDK, PydanticAI, CrewAI, AutoGen), the evaluation tooling hasn't caught up. MLflow has some agent eval capabilities, and Braintrust/Langfuse can trace agents, but nobody is doing:

- CI-gated agent regression testing
- Trajectory quality evaluation (not just outcome)
- Constraint enforcement (tool call budgets, required/forbidden tools)
- Agent-specific model migration

This is Scaffold's biggest opportunity for differentiation. The market is unserved.

---

## Go-to-Market Implications

### Primary positioning
**"CI-native regression testing for LLMs. Provider-neutral. No platform required."**

### Target personas
1. **ML/AI engineers** who maintain LLM-powered features and are tired of manual testing
2. **Platform/infra engineers** who want to standardize LLM testing across teams
3. **Teams migrating off Promptfoo** due to the OpenAI acquisition

### Key messages by competitor

| When competing against | Lead with |
|---|---|
| **Promptfoo** | Provider neutrality, model migration automation, pipeline-level testing, agentic eval |
| **Braintrust** | Zero infrastructure, no platform lock-in, open source, migration automation |
| **Langfuse** | Purpose-built for CI (not observability), migration automation, agentic eval |
| **DeepEval** | Pipeline-level testing (not just prompt), migration automation, language-agnostic command mode |
| **DSPy** | CI integration, holdout-validated migration, agentic trajectory eval, not a framework rewrite |
| **"We don't test"** | Five-minute setup, catches regressions before users do, model migration without the fire drill |

### Wedge use cases (easiest adoption paths)
1. **Model migration** — Team needs to upgrade a model and wants automated prompt re-tuning. Immediate, concrete value.
2. **Post-Promptfoo migration** — Team is looking for alternatives after the OpenAI acquisition. Drop-in replacement with more features.
3. **First CI gate** — Team has no LLM testing at all and wants the simplest possible setup. `scaffold init` → `scaffold run` in five minutes.
