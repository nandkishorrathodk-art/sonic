# AVCS-LLM Goal Context

**Status:** Approved design target, 2026-10-23. This document is the binding
goal for the AVCS/computer-use work that follows. It describes what to build,
not what currently works. Until the build is complete and measured, treat every
claim here as a target, not a capability.

**Approved by the operator on 2026-10-23.** The operator confirmed:

1. AVCS becomes its own **cloud vision LLM** (the local SmolVLM is removed from
   the plan).
2. **The brain LLM never receives an image.** Not one pixel. It receives only
   AVCS's text description of what is happening.
3. AVCS **streams in real time** — it watches the screen continuously.
4. AVCS tells the brain, in words, what is on screen and what changed.
5. The brain decides what to do next and tells AVCS in words.
6. AVCS finds the thing, acts on it, watches the result, and reports back.
7. The **reflex layer and the button/affordance detector are removed from the
   active path.** AVCS must understand the screen, not count pixel changes.

---

## 1. The two-model contract

There are exactly two reasoning models. Their split is strict.

| Role | Model | Input | Output |
|---|---|---|---|
| **Brain** | text reasoning LLM | text only: the goal, AVCS's description, the app list, action history | **one instruction, in words** |
| **AVCS** | cloud vision LLM | the **live screenshot** + the instruction | **the action + what it observed** |

Hard rules:

- The brain **never** receives an image. If a code path would attach a
  screenshot to the brain's request, that path is wrong and must fail closed.
- AVCS **always** looks at pixels. It is the only component allowed to.
- The brain speaks in intent ("click the Clone button"), never in coordinates.
- AVCS speaks in observations ("a dropdown appeared with HTTPS and SSH clone
  URLs"), never in raw pixel statistics.

## 2. The loop

```
        USER task
            │
            ▼
    ┌───────────────┐
    │  Brain (LLM)  │  text only — never sees pixels
    │  decides WHAT │
    └───────┬───────┘
            │  instruction (words)
            ▼
    ┌───────────────┐
    │  AVCS (cloud  │  sees the live screen
    │  vision LLM)  │  understands it
    │  decides WHERE│  finds the target
    │  and ACTS     │  performs the action
    └───────┬───────┘
            │  observation (words)
            ▼
    ┌───────────────┐
    │  Brain        │  replan
    └───────┬───────┘
            │  next instruction
            └──────────► back to AVCS
                         ...
                         task done → report to user
```

## 3. The wire format

Brain to AVCS:

```json
{ "instruction": "click the Clone button", "expect": "a clone dialog opens" }
```

AVCS to Brain:

```json
{
  "action_taken": "click",
  "point": [1512, 640],
  "observation": "A dropdown appeared with HTTPS and SSH clone URLs",
  "confidence": 0.91
}
```

When AVCS cannot find the target it must say so, not guess:

```json
{ "action_taken": "none", "observation": "No element matching 'Clone button' is visible" }
```

## 4. Real-time streaming

- **Frames stream continuously.** AVCS always holds a current view of the
  screen, so when an instruction arrives it acts on what is on screen now.
- **AVCS-LLM runs per step** — on each instruction, and after each action to
  check the result.
- "Real-time" applies to *watching*. *Acting* is per-step.

Honest latency on this machine (CPU-only, no GPU, cloud vision model):

| Operation | Time |
|---|---|
| Capture a frame | ~100 ms |
| AVCS cloud call (see + act) | ~3-6 s |
| Brain call (decide) | ~1-2 s |
| **Per step** | **~4-8 s** |

A cloud VLM cannot stream at 10-30 fps. That is a physical limit, not an
engineering gap. Genuine 30 fps vision needs a GPU.

## 5. Concrete trace — the operator's example

Task: *"open the file manager, create file.txt, then open Firefox, search
GitHub for a repo, and clone it."*

**Step 0 — first look**
- AVCS captures a frame and describes it: *"Desktop, no windows open. Icons
  visible: Files, Firefox, Terminal."*
- Brain reads that text: **"Launch the file manager."** (a typed app-launch
  action, no pixels involved)

**Step 1 — create the file**
- AVCS: *"A file manager window is open, showing the Home folder."*
- Brain: *"Create a new empty file named file.txt."*
- AVCS (right-clicks the empty area): *"A context menu opened: Open, Create
  Document, Paste."*
- Brain: *"Choose Create Document, then Empty File."*
- AVCS (clicks): *"A name field appeared with 'untitled' selected."*
- Brain: *"Type file.txt and press Enter."*
- AVCS (types, presses Enter): *"A new file named file.txt now exists."*
- Brain: *"file.txt created. Next: open Firefox."*

**Step 2 — Firefox and GitHub**
- Brain: *"Launch Firefox."* → AVCS: *"Firefox window open, homepage showing."*
- Brain: *"Click the address bar and go to github.com."* → AVCS clicks, types,
  Enter: *"GitHub homepage loaded."*
- Brain: *"Search for 'openhands sdk'."* → AVCS: *"Search results page showing
  repositories."*
- Brain: *"Click the first repository result."* → AVCS: *"Repository page:
  openhands/software-agent-sdk. There is a green 'Code' button."*
- Brain: *"Click the Code button."* → AVCS: *"A dropdown appeared with HTTPS
  and SSH clone URLs."*

**Step 3 — clone**
- Brain: *"Open a terminal and clone the HTTPS URL."*
- The **terminal plane** (not AVCS) runs `git clone ...` → *"Cloning into
  'software-agent-sdk'... done."*
- Brain: *"Task complete."*

**Report to user**
> "Opened the file manager and created file.txt. Opened Firefox, searched
> GitHub for 'openhands sdk', opened openhands/software-agent-sdk, and cloned
> it into ~/software-agent-sdk."

## 6. What is removed from the active path

The operator asked for these to go:

| Removed | Reason |
|---|---|
| Reflex pixel-change layer as the perception source | AVCS must understand the screen, not count changes |
| Affordance / "clickable center" head | A human sees *things* ("the Clone button"), not "clickable centers" |
| Local SmolVLM describer | Too slow (~19 s) and cannot locate elements |

Removal means: not consulted for perception or grounding. Any surviving code
must not be presented as the perception path.

## 7. What replaces them

**AVCS cloud adapter** — one module with one contract:

```
input:  screenshot (RGB/PNG) + instruction (text)
output: action (click | type | key | scroll | none)
        + point/text (when applicable)
        + observation (text)
        + confidence
```

Backed by a cloud vision LLM. The operator chose cloud over a local trained
model, accepting the tradeoff: it works sooner, but it is a cloud service, not
a model we own, and every step needs the network and an API key.

Verified on this machine before approval (so the choice is evidence-based):

| Model | Localize a named UI element? |
|---|---|
| SmolVLM-500M (local) | No — invents fake JSON, no coordinates |
| llama-3.2-11b-vision | No — returns JSON but wrong coordinates |
| **nemotron-3.5-super-vl-120b** | **Yes — ~1.3-1.9% error on a test UI** |

## 8. Honest limits — do not hide these

1. **Typing is the risk.** Creating "file.txt" and typing "github" require a
   *type* action, not a click. A cloud VLM can emit it, but it is less reliable
   than clicking and may need retries.
2. **Real-desktop accuracy is unproven.** The ~1.5% figure is from a clean
   640x400 test UI. A real 1920x1080 desktop with dense UI is unmeasured. It
   must be measured before any accuracy claim.
3. **Cloud dependency.** Every AVCS step needs the network and the API key. If
   either drops, the loop stalls.
4. **Not our own model.** This is a cloud service, not a trained AVCS model.
5. **Per-step latency is seconds**, not milliseconds.

## 9. Build plan

1. **AVCS cloud adapter** — screenshot + instruction → cloud VLM → action +
   observation, with strict output parsing and fail-closed on unparseable or
   not-found responses.
2. **Wire it into the agent** — semantic click/type targets route through the
   AVCS adapter; the brain still receives text only.
3. **Continuous frame stream** — keep the live capture so AVCS always acts on
   the current screen.
4. **Remove the reflex / affordance path** from the active perception loop.
5. **Observation feedback** — AVCS's observation goes back to the brain each
   step for replanning.
6. **Honest capability surface** — the operator sees whether AVCS is ready and
   the measured accuracy.

## 10. Definition of done

The design is met only when:

- A GUI action on a described control resolves to real pixels via AVCS.
- AVCS reports an observation the brain can replan from.
- **No screenshot reaches the brain** — proven by a test, not asserted.
- Reflex and the affordance head are not consulted for perception.
- Accuracy is measured on a real desktop and reported honestly, including
  failure.

Unit tests alone do not satisfy this. A live, isolated desktop task with real
receipts does.
