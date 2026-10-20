# No-Chiting: Honest SONIC Computer Interaction

SONIC is the operator of its workstation. The assistant must not impersonate
SONIC, operate the desktop behind SONIC, or present an API/provider shortcut as
visible mouse or keyboard work.

## Required behavior

- Every requested computer action must be performed by SONIC's own
  Boss-to-`ComputerUseAgent` execution path.
- Do not directly click, move the pointer, type, launch, close, or otherwise
  manipulate workstation applications to simulate SONIC's progress.
- `APP_LAUNCH` and `APP_CLOSE` are application-level operations. They do not
  demonstrate cursor movement or a visible mouse click. Describe their actual
  action receipts accurately.
- A claimed pointer movement or click requires a SONIC GUI-action receipt and
  independent evidence from the live workstation that the action was dispatched.
  A cursor observed at a single position does not prove it moved.
- GUI click targets require a fresh frame and a configured, evaluated local AVCS
  grounder. If grounding is missing, invalid, ambiguous, or stale, block the
  click. Never substitute guessed coordinates, application names, direct
  provider calls, shell commands, or manual assistant interaction.
- A dispatched action is not proof of task completion. Verify the requested
  postcondition from fresh workstation evidence and report failure or
  inconclusive results honestly.
- Stop after a capability block or action-loop breaker. Do not retry the same
  action repeatedly or describe a blocked/recovered action as success.

## Reporting

For computer-use tasks, report separately:

1. What SONIC proposed.
2. What action SONIC actually dispatched, if any.
3. What the live workstation independently showed.
4. Whether the requested outcome was verified.

If the required local grounding or focus-verification model is unavailable,
state that cursor-driven GUI interaction is currently blocked. Do not claim
human-grade or arbitrary-app capability from an application launch, unit test,
or screenshot alone.
