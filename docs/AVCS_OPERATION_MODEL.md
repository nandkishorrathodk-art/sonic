# AVCS Operation Model — Who Operates Whom

**Status:** Binding clarification, 2026-10-23. This document supersedes the
"AVCS daemon acts on its own" wording in `AVCS_LLM_GOAL_CONTEXT.md`. Where the
two documents disagree about who dispatches GUI actions, **this document wins.**

**Operator instruction that this document records:** *"I want SONIC to operate
it. SONIC should be the agent. AVCS is only real-time perception."*

---

## 1. The one-line rule

**SONIC operates AVCS. AVCS never operates SONIC, and AVCS never acts on the
desktop by itself.**

AVCS is an eye. SONIC is the brain and the hand.

---

## 2. Correction to the earlier goal document

`AVCS_LLM_GOAL_CONTEXT.md` (approved 2026-10-23) described a daemon that
performed its own mouse and keyboard actions:

```
[AVCS DAEMON]
   ├── 3. Hands (Actuator): Moves mouse -> Clicks -> Types "github.com"
```

That is **not** the architecture we are building. The corrected model:

| Question | Earlier wording | This document (binding) |
|---|---|---|
| Who decides what to do? | Brain LLM | **SONIC agent** |
| Who dispatches clicks/typing? | AVCS daemon | **SONIC agent** |
| What is AVCS? | Eyes **and** hands | **Eyes only** |
| Does AVCS ever act alone? | Yes | **Never** |

The code already works the corrected way. This document aligns the written
goal with the implemented reality.

---

## 3. Verified control direction (from the code)

AVCS is a service that SONIC calls. The call direction is one-way:

| SONIC calls | AVCS returns | Source |
|---|---|---|
| `_ground_target_with_avcs(query)` | a screen point, or nothing | `computer_use/agent.py:6119`, `:6963` |
| `_verify_gui_focus_with_avcs(...)` | a focus verdict | `computer_use/agent.py:7498` |

AVCS's own awareness loop is documented as observation-only:

> *"It never acts. Awareness only observes and publishes. No GUI action is
> dispatched from this task, so it cannot fight an agent that owns the
> desktop."*
> — `computer_use/awareness.py:16`

AVCS publishes perception events. It does not subscribe to its own output and
it does not move the pointer.

---

## 4. Roles, restated as System 1 / System 2

| Role | Component | Responsibility | Never does |
|---|---|---|---|
| **System 2 — Brain** | SONIC agent reasoning (text LLM) | Interpret the goal, plan, choose the next action, replan | Never receives pixels |
| **System 1.5 — Eye that points** | AVCS grounding (UI-TARS or a trained local model) | Turn a named target into screen pixels | Never decides, never clicks |
| **System 1 — Eye that notices** | AVCS reflex | Report frame/scene change and motion | Never identifies a control by name |
| **Hand** | SONIC computer provider | Dispatch the click/typing the agent chose | Never chooses the action |
| **Check** | Boss / Judge + AVCS diff | Confirm the postcondition from fresh evidence | Never infers success from a scene change alone |

The brain is **text-only**. It never sees an image. It reads AVCS's text
description and issues the next instruction.

---

## 5. The full loop, step by step

```
[USER] "create file.txt, then clone the OpenHands SDK"
   |
   v
+------------------------------------------------------+
| SONIC AGENT  (operator + brain)                      |
|   - plans the task                                   |
|   - asks AVCS where things are                       |
|   - dispatches clicks and typing itself              |
|   - verifies, then replans                           |
+------+--------------------------------+--------------+
       | 1. where is X?                 | 4. click/type at (x,y)
       v                                v
+---------------------+        +------------------------+
| AVCS (perception)   |        | SONIC provider (motor) |
|  - reflex  ~19 ms   |        |  - smooth cursor glide |
|  - ground  1-3 s    |        |  - xdotool click       |
|  - verify           |        |  - xdotool type        |
+---------------------+        +------------------------+
       | 2. "Clone is at (1512, 640)"   | 5. "the screen changed"
       +------------> SONIC <-----------+
```

Every step:

1. **SONIC** decides: "click the Clone button".
2. **SONIC** asks AVCS: where is the Clone button?
3. **AVCS** answers: at pixel (1512, 640). It does nothing else.
4. **SONIC** dispatches the click through its own computer provider.
5. **AVCS** reports the effect: the screen changed.
6. **SONIC** verifies and replans the next step.

AVCS appears at steps 2-3 and step 5. It never appears as the actor.

---

## 6. Latency reality

The perception and reasoning layers are not the same speed, and should not be
described as if they were.

| Layer | Typical latency | Real-time? |
|---|---|---|
| AVCS reflex | ~19 ms | Yes |
| AVCS grounding (cloud VLM) | ~1-3 s per call | No |
| SONIC brain (text LLM) | ~1-5 s per turn | No |

The grounding model is a **per-step tool**, not a real-time stream. Only the
reflex layer is real-time. The agent's loop is therefore paced by reasoning and
grounding, not by the frame rate.

---

## 7. What "AVCS is only real-time perception" means concretely

| AVCS does | AVCS does not |
|---|---|
| Capture frames | Decide the next action |
| Report scene/change/motion | Choose which control to use |
| Answer "where is X?" | Click or type |
| Verify a focus patch | Plan a multi-step task |
| Publish perception events | Own the desktop |

---

## 8. What SONIC keeps doing

| SONIC does | Where |
|---|---|
| Decide and plan | `computer_use/agent.py:choose_action` |
| Ask AVCS for a target | `_ground_target_with_avcs` |
| Dispatch clicks | `computer/docker_computer.py:gui_action` |
| Dispatch typing | `computer/docker_computer.py` + `computer_use/motor.py` |
| Verify with AVCS | `_verify_gui_focus_with_avcs` |
| Replan on failure | agent replan loop |

---

## 9. The two missing capability slots

Everything above is wired except two models. Both are perception inputs to
SONIC; neither is an actor.

| Slot | Purpose | Status | Who consumes it |
|---|---|---|---|
| Grounding model | "where is X?" | Not trained / not wired | SONIC agent |
| Focus head | "is the cursor in a text field?" | Not trained | SONIC agent |

When both exist, SONIC can click a named control and type into a verified
field — using its own provider, under its own control.

---

## 10. Summary

- **SONIC operates AVCS.** Not the reverse.
- **AVCS is eyes only** — it perceives, points, and verifies. It never acts.
- **The brain is text-only** and never receives a pixel.
- **The grounding model is a per-step tool**, not a real-time loop.
- **Two perception slots remain empty:** grounding and focus. Filling them is
  the whole remaining GUI project.
