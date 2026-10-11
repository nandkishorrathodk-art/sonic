# AVCS SDK v0.1 — Contract

## Lifecycle

A client must be connected before observe, target discovery or action execution. Closing a client releases the backend session and invalidates client-held observations.

## Observation and target validity

Observations are timestamped and use `normalized_virtual_desktop` coordinates. Target bounding boxes and points must be finite and normalized. Targets are tied to an `observation_id`; coordinate-based actions must provide the `observation_id` used to derive the point. Target-based and coordinate-based actions are rejected if the observation is unknown, expired, evicted or is not the latest observation. A concrete backend must map normalized coordinates to physical desktop pixels using the exact geometry snapshot associated with the observation.

## Action results

Terminal statuses are `confirmed`, `failed`, `ambiguous`, `blocked` and `cancelled`. `confirmed` requires `VerificationEvidence(verified=True, ...)` demonstrating the expected postcondition. Pixels changing or an OS input API accepting an event alone are not sufficient verification.

If the backend cannot establish whether dispatch occurred, it must return/raise an ambiguous result. The client must not retry it. Use the same `action_id` to join an in-flight operation or retrieve the cached result within the same client instance. Caller cancellation does not cancel the shielded backend operation; a same-ID request joins it instead of dispatching twice. This in-memory cache is not a durable cross-process exactly-once mechanism.

## Idempotency and risk

`ActionClass` is declared by the caller as context for broker policy; it is not proof that an operation is safe to repeat. The real broker must enforce repeat behavior. High-risk actions require an `authorization_ref`; the broker must validate that reference against trusted state, bind it to the action and expiry, and refuse self-issued or stale grants.

## Compatibility

The SDK uses dataclasses and enums, with no runtime dependencies, and targets Python 3.10+. v0.1 is pre-release. Breaking changes are permitted before v1.0; serialized wire-format compatibility is not promised in this scaffold.
