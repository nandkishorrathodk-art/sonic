# SDK release checklist

- [ ] Adapter implemented against the actual `ActionBroker`, `HumanEye`, capture adapter and verifier APIs.
- [ ] Contract tests run against the real repository backend.
- [ ] Action dispatch always routes through the broker; no parallel raw-input path exists.
- [ ] Target and observation freshness enforced again at broker dispatch.
- [ ] `CONFIRMED` requires independently verified postcondition evidence.
- [ ] Ambiguous dispatch never triggers automatic retry.
- [ ] Authorization references validated by trusted broker state for high-risk actions.
- [ ] Cancellation, shutdown and input-session loss tested.
- [ ] Logs redact screenshots, typed text and accessibility content by default.
- [ ] Supported Windows builds, Python versions, DPI configurations and capture backends documented.
- [ ] Build wheels and install them into a clean Windows virtual environment.
- [ ] Test source install and wheel install; verify package contents and import paths.
- [ ] Add API compatibility tests and dependency/license/security scans in CI.
- [ ] Choose and include the approved repository license; do not publish with placeholder licensing metadata.
- [ ] Publish versioning/deprecation policy, security contact, and changelog.
- [ ] Tag and sign release after reviewing benchmark and regression artifacts.
