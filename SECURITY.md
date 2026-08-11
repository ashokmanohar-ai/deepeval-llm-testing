# Security Policy

## Reporting a vulnerability

Do not open a public issue containing credentials, private prompts, retrieved documents,
customer data, or exploit details. Report sensitive findings privately to the repository
owner through GitHub's security-advisory feature.

## Credential rules

- Keep `.env` and `.env.local` untracked.
- Store Azure credentials in a secret manager or GitHub Actions secrets.
- Use a dedicated Azure deployment with least-privilege access and budget controls.
- Rotate a key immediately if it appears in source, logs, an issue, or a test artifact.
- Treat prompts, outputs, contexts, traces, and evaluation reasons as potentially sensitive.
- Do not enable optional cloud result upload for regulated data without an approved review.

The sample dataset is fictional and contains no credentials or personal information.

