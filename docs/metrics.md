# Metric guide

This guide uses DeepEval 4.1.7 terminology and API names.

## Answer relevancy

- **Measures:** whether statements in `actual_output` address `input`.
- **Requires:** `input`, `actual_output`.
- **Conceptual score:** relevant statements divided by total statements.
- **Interpretation:** higher is better; the score must be at least the threshold.
- **Typical use:** chat, search answers, RAG generator output, support assistants.
- **Common mistake:** treating a relevant answer as necessarily correct. A confident false answer
  may be relevant but factually wrong.
- **Starting threshold:** 0.7–0.8, then calibrate.

## Faithfulness

- **Measures:** whether answer claims align with the content actually retrieved by a RAG pipeline.
- **Requires:** `input`, `actual_output`, `retrieval_context`.
- **Conceptual score:** supported claims divided by claims in the answer.
- **Interpretation:** higher is better.
- **Typical use:** isolating the quality of a RAG generator after retrieval.
- **Common mistake:** supplying ideal documents as `retrieval_context`. Use what the retriever
  actually returned, including imperfect results.
- **Starting threshold:** 0.8 for grounded business answers; raise it for high-risk domains.

## Hallucination

- **Measures:** contradictions between `actual_output` and curated ground-truth `context`.
- **Requires:** `input`, `actual_output`, `context`.
- **Conceptual score:** contradicted contexts divided by all supplied contexts.
- **Interpretation:** **lower is better**; the score must be at or below the threshold.
- **Typical use:** regression tests with reviewed reference facts.
- **Common mistake:** applying a normal higher-is-better threshold. A hallucination score of 0 is
  ideal and 1 is worst.
- **Starting threshold:** 0.2–0.3; use 0 in strict, high-confidence cases.

## G-Eval

- **Measures:** any clearly stated custom quality policy, such as correctness, completeness,
  professionalism, clarity, helpfulness, tone, or response quality.
- **Requires:** `input`, `actual_output`, plus every parameter named by the criteria.
- **Conceptual score:** an LLM judge follows generated or supplied evaluation steps and normalizes
  its rating to 0–1.
- **Interpretation:** higher is better, but results are nondeterministic.
- **Typical use:** domain-specific rubrics for which exact string matching is too rigid.
- **Common mistake:** including `SingleTurnParams` not mentioned in the criteria, or combining too
  many unrelated qualities into one opaque score.
- **Starting threshold:** 0.7–0.8 after comparing judge results with human labels.

## Tool correctness

- **Measures:** whether the agent called the expected tools, optionally including arguments,
  outputs, and ordering.
- **Requires:** `input`, `actual_output`, `tools_called`, `expected_tools`.
- **Conceptual score:** correctly used calls divided by calls. It is deterministic unless
  `available_tools` is supplied for LLM-judged optimality.
- **Interpretation:** higher is better; use 1.0 when the tool contract is exact.
- **Common mistake:** evaluating only the final prose and missing a risky or wasteful tool call.

DeepEval 4.1.7 initializes a model object when constructing this metric even when only the
deterministic comparison is used. This repository supplies a fail-closed no-op model for that path;
it cannot call a provider and raises if DeepEval unexpectedly requests a judge.

## Task completion

- **Measures:** alignment between the requested task and the outcome of the full agent trace.
- **Requires:** DeepEval tracing and an observed multi-step run.
- **Conceptual score:** an LLM-judged alignment between inferred or explicit task and outcome.
- **Interpretation:** higher is better.
- **Typical use:** end-to-end agents that plan, call tools, hand off, and synthesize outcomes.
- **Common mistake:** trying to use it as a plain `LLMTestCase` metric. In DeepEval 4.1.7 it is a
  trajectory metric and needs tracing.

## Choosing thresholds

Do not select production thresholds from these examples alone.

1. Assemble representative passing, borderline, and failing outputs.
2. Have qualified reviewers label them independently.
3. Run the metric several times when it uses an LLM judge.
4. Compare false-pass and false-fail rates at candidate thresholds.
5. Choose the operating point that matches business risk.
6. Version the dataset, metric prompt, judge deployment, and threshold together.
7. Monitor drift when the application model, judge model, or source data changes.

For nondeterministic gates, consider repeated runs, score margins, a review band, and a separate
deterministic policy layer for rules that must never vary.
