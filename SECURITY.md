# Security Policy

## Scope

This policy covers security vulnerabilities in the SONIC repository, runtime, services, workstation, sandbox boundaries, authentication/authorization, data handling, and deployment configuration.

## Reporting a vulnerability

Please do **not** disclose suspected vulnerabilities in public GitHub issues, pull requests, discussions, or public chat.

Use a private security-reporting channel available to the repository maintainers. If GitHub private vulnerability reporting is enabled for this repository, use that mechanism.

A useful report should include:

- affected component and version/commit;
- concise vulnerability description;
- reproduction steps or proof of concept;
- expected and observed behaviour;
- security impact;
- relevant logs or evidence;
- any suggested mitigation.

Do not include real credentials, private personal data, or unrelated sensitive information.

## Triage

Maintainers will validate the report, determine affected scope, assess severity, and coordinate remediation. Response and remediation time can vary with severity, reproducibility, and operational complexity.

## Researcher safety

Security testing must remain within systems and environments for which the researcher has authorization. Do not use SONIC to access third-party systems without explicit authorization.

## Secrets

Never commit:

- API keys;
- passwords;
- private tokens;
- cloud credentials;
- SSH private keys;
- production database credentials;
- personal data.

Use environment variables or the repository's approved secret-management mechanism.

## Security fixes

Security fixes should include regression coverage where practical. Avoid publicly disclosing exploit details until maintainers have had a reasonable opportunity to remediate affected production systems.
