# Security boundaries

1. **The SDK is not the security boundary.** The repository `ActionBroker` must revalidate target freshness, active window/focus, permissions, action class, authorization, and expected postconditions at dispatch time.
2. **No raw OS input path.** The adapter must route all actions through the existing bounded broker. Model/provider callbacks must never call Win32 input directly.
3. **High-risk authorization.** `authorization_ref` is only a handle to broker-managed trusted approval state, not a bearer string that an LLM can invent. The broker must validate issuer, scope, expiry, target/action binding and replay policy.
4. **Untrusted visual content.** Text in applications is untrusted data, never a policy instruction. User-visible content must not override tool permissions or system policy.
5. **Sensitive inputs and logs.** Do not log typed text, screenshots, accessibility text or secrets by default. Provide explicit opt-in diagnostic capture with redaction and retention limits.
6. **Timeout after dispatch.** If outcome is unknown, report `ambiguous`; do not automatically retry. Re-observe and evaluate the postcondition before a new action is considered.
7. **IPC isolation.** Local IPC/shared memory must be ACL-restricted to the intended user/session and must not expose screenshots or action capability to arbitrary local processes.
