Privacy & Security Design

- Data Minimization: Store only necessary PII (email, name); keep most analytics de-identified. Never expose secrets to the client.
- Consent & Transparency: Obtain explicit consent for data collection; provide clear policy and allow data export/deletion on request.
- Encryption: TLS in transit; at-rest encryption for DB backups and PII columns where applicable.
- Access Control: Role-based access (student, educator, admin); least privilege; audit logs for admin actions and data access.
- Retention: Define retention periods for raw logs; aggregate analytics after N days; purge per user request.
- Security Hygiene: Secret management (env vars, vault), dependency updates, vulnerability scanning, and periodic penetration testing.
- LLM Safety: Never send unnecessary PII to LLM; redact where possible; prefer grading/generation prompts that avoid user identifiers.
- Compliance Readiness: Align with FERPA-like educational data protections; adhere to GDPR principles for EU users (lawful basis, rights, DPIA when needed).

