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

The operator supplied this architecture directly:

```
[USER] "file.txt banao aur openhands sdk clone karo"
   │
   ▼
[BRAIN LLM] (Pure Text Strategist — Fast Text Model)
   │  Out: {"instruction": "Click the address bar and type github.com", "expect": "GitHub loads"}
   ▼
[AVCS DAEMON] (Cloud Vision Stream + Local Mouse/Keyboard Controller)
   ├── 1. Eyes (Stream): Reads latest 1080p frame from live ring buffer
   ├── 2. Grounding: Finds Address Bar at normalized [240, 82] -> real [460, 88]
   ├── 3. Hands (Actuator): Moves mouse -> Clicks -> Types "github.com" -> Hits Enter
   ├── 4. Stream Watch: Watches frames until page load finishes (or timeout)
   └── 5. Report: Generates text observation of new screen state
   │
   │  Out: {"action_taken": "click_and_type", "point": [460, 88], "observation": "GitHub home page is loaded with a search bar in the top right", "confidence": 0.95}
   ▼
[BRAIN LLM] -> plans the next step -> ... -> task done -> report to user
```

---

## 1. The two-model contract

There are exactly two components. Their split is strict.

| Role | Component | Input | Output |
|---|---|---|---|
| **Brain** | Pure Text Strategist — a **fast text model** | text only: the goal, AVCS's observation, the app list, action history | **one instruction, in words** |
| **AVCS** | **daemon**: cloud vision LLM + local mouse/keyboard actuator | the **live frame** from the ring buffer + the instruction | **the action it took + what it observed** |

Hard rules:

- The brain **never** receives an image. If a code path would attach a
  screenshot to the brain's request, that path is wrong and must fail closed.
- AVCS **always** looks at pixels. It is the only component allowed to.
- The brain speaks in intent ("click the address bar and type github.com"),
  never in coordinates.
- AVCS speaks in observations ("GitHub home page is loaded with a search bar in
  the top right"), never in raw pixel statistics.
- AVCS is a **daemon**, not a per-call helper: it owns the frame stream, the
  grounding model, and the mouse/keyboard actuator, and it keeps running between
  instructions.

## 2. The AVCS daemon internals

The daemon is one long-lived service with five internal stages. A single
instruction drives one pass through them.

```
[AVCS DAEMON]  (Cloud Vision Stream + Local Mouse/Keyboard Controller)
   ├── 1. Eyes (Stream)     : read the latest 1080p frame from the live ring buffer
   ├── 2. Grounding         : find the target, normalized [240, 82] -> real [460, 88]
   ├── 3. Hands (Actuator)  : move mouse -> click -> type -> press Enter
   ├── 4. Stream Watch      : watch frames until the page/state settles (or timeout)
   └── 5. Report            : turn the new screen state into a text observation
```

| Stage | Owns | Never does |
|---|---|---|
| Eyes | a live ring buffer of recent frames, always current | interpret pixels |
| Grounding | the cloud vision LLM call; intent -> pixel point | dispatch input |
| Hands | the local mouse/keyboard actuator on `DISPLAY` | decide what to do |
| Stream Watch | settling detection after an action (frame stability / timeout) | claim success from a single frame |
| Report | the text observation the brain replans from | send pixels anywhere |

Stage 3 is why typing is AVCS's job: the daemon owns the actuator, so a compound
action such as `click_and_type` (focus the field, type, press Enter) happens
inside AVCS, not in the brain.

## 3. The loop

```
[USER] "file.txt banao aur openhands sdk clone karo"
   │
   ▼
[BRAIN LLM]  (Pure Text Strategist — fast text model)
   │  Out: {"instruction": "Click the address bar and type github.com",
   │        "expect": "GitHub loads"}
   ▼
[AVCS DAEMON]  (Cloud Vision Stream + Local Mouse/Keyboard Controller)
   ├── 1. Eyes (Stream)   : latest 1080p frame from the live ring buffer
   ├── 2. Grounding       : address bar at normalized [240, 82] -> real [460, 88]
   ├── 3. Hands (Actuator): move mouse -> click -> type "github.com" -> Enter
   ├── 4. Stream Watch    : watch frames until the page load settles (or timeout)
   └── 5. Report          : text observation of the new screen state
   │
   │  Out: {"action_taken": "click_and_type", "point": [460, 88],
   │        "observation": "GitHub home page is loaded with a search bar in
   │                        the top right", "confidence": 0.95}
   ▼
[BRAIN LLM] -> plans the next step -> ... -> task done -> report to user
```

The brain never sees a frame. It reads AVCS's observation as text and replies
with the next instruction as text. AVCS is the only component that touches
pixels, the mouse, or the keyboard.

## 4. The wire format

Brain to AVCS:

```json
{ "instruction": "Click the address bar and type github.com", "expect": "GitHub loads" }
```

AVCS to Brain:

```json
{
  "action_taken": "click_and_type",
  "point": [460, 88],
  "observation": "GitHub home page is loaded with a search bar in the top right",
  "confidence": 0.95
}
```

`action_taken` is one of `click`, `double_click`, `right_click`, `click_and_type`,
`type`, `key`, `scroll`, `drag`, or `none`. A compound action such as
`click_and_type` is performed entirely inside the daemon (focus, type, Enter).

When AVCS cannot find the target it must say so, not guess:

```json
{ "action_taken": "none", "observation": "No element matching 'Clone button' is visible" }
```

## 5. Real-time streaming

- **Frames stream continuously** into the daemon's ring buffer. AVCS always
  holds a current view of the screen, so when an instruction arrives it acts on
  what is on screen *now*, not on a stale frame.
- **AVCS runs one pass per instruction**, and again after the action to watch
  the result settle (stage 4, Stream Watch).
- "Real-time" applies to *watching* (the ring buffer stays current). *Acting*
  is per-step.

Honest latency on this machine (CPU-only, no GPU, cloud vision model):

| Operation | Time |
|---|---|
| Capture a frame | ~100 ms |
| AVCS cloud call (ground + decide) | ~3-6 s |
| Brain call (decide) | ~1-2 s |
| **Per step** | **~4-8 s** |

A cloud VLM cannot stream at 10-30 fps. That is a physical limit, not an
engineering gap. Genuine 30 fps vision needs a GPU. The ring buffer makes the
*frame* real-time; the *vision model* is per-step.

## 6. Concrete trace — the operator's example

Task: *"file.txt banao aur openhands sdk clone karo"* — create file.txt and
clone the openhands sdk.

**Step 0 — first look**
- AVCS (Eyes + Report): *"Desktop, no windows open. Icons visible: Files,
  Firefox, Terminal."*
- Brain reads that text: **"Launch the file manager."** (typed app-launch, no
  pixels involved)

**Step 1 — create the file**
- AVCS: *"A file manager window is open, showing the Home folder."*
- Brain: *"Create a new empty file named file.txt."*
- AVCS (right-clicks the empty area): *"A context menu opened: Open, Create
  Document, Paste."*
- Brain: *"Choose Create Document, then Empty File."*
- AVCS (clicks): *"A name field appeared with 'untitled' selected."*
- Brain: *"Type file.txt and press Enter."*
- AVCS (`click_and_type`): *"A new file named file.txt now exists."*
- Brain: *"file.txt created. Next: clone openhands sdk."*

**Step 2 — Firefox and GitHub**
- Brain: *"Launch Firefox."* → AVCS: *"Firefox window open, homepage showing."*
- Brain: *"Click the address bar and type github.com."* → AVCS
  (`click_and_type`): *"GitHub homepage loaded."*
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
> "Created file.txt in the home folder. Opened Firefox, searched GitHub for
> 'openhands sdk', opened openhands/software-agent-sdk, and cloned it into
> ~/software-agent-sdk."

## 7. What is removed from the active path

The operator asked for these to go:

| Removed | Reason |
|---|---|
| Reflex pixel-change layer as the perception source | AVCS must understand the screen, not count changes |
| Affordance / "clickable center" head | A human sees *things* ("the Clone button"), not "clickable centers" |
| Local SmolVLM describer | Too slow (~19 s) and cannot locate elements |

Removal means: not consulted for perception or grounding. Any surviving code
must not be presented as the perception path.

## 8. What replaces them

**AVCS daemon** — one long-lived service with one contract:

```
input:  live frame (from the ring buffer) + instruction (text)
output: action_taken (click | double_click | right_click | click_and_type
                      | type | key | scroll | drag | none)
        + point (pixels) and/or text (for typing)
        + observation (text)
        + confidence
```

Internally the daemon runs the five stages of section 2: Eyes, Grounding,
Hands, Stream Watch, Report. Grounding is backed by a cloud vision LLM.

The operator chose cloud over a local trained model, accepting the tradeoff: it
works sooner, but it is a cloud service, not a model we own, and every step
needs the network and an API key.

Verified on this machine before approval (so the choice is evidence-based):

| Model | Localize a named UI element? |
|---|---|
| SmolVLM-500M (local) | No — invents fake JSON, no coordinates |
| llama-3.2-11b-vision | No — returns JSON but wrong coordinates |
| **nemotron-3.5-super-vl-120b** | **Yes — ~1.3-1.9% error on a test UI** |

The local **actuator** (mouse/keyboard on `DISPLAY`) is not a cloud call: it is
the existing computer provider, driven by the daemon after grounding.

## 9. Honest limits — do not hide these

1. **Typing is the hardest part.** Creating "file.txt" and typing "github"
   require a *type* action, not a click. AVCS owns typing (the daemon has the
   actuator), but a cloud VLM is less reliable at emitting exact text than at
   locating a control, and it may need retries. Grounding and typing are
   different capabilities; do not assume one implies the other.
2. **Real-desktop accuracy is unproven.** The ~1.5% figure is from a clean
   640x400 test UI. A real 1920x1080 desktop with dense UI is unmeasured. It
   must be measured before any accuracy claim.
3. **Cloud dependency.** Every AVCS step needs the network and the API key. If
   either drops, the loop stalls.
4. **Not our own model.** This is a cloud service, not a trained AVCS model.
5. **Per-step latency is seconds**, not milliseconds.

## 10. Build plan

1. **AVCS daemon skeleton** — a long-lived service holding the frame ring
   buffer (Eyes), the cloud grounding call (Grounding), the local actuator
   (Hands), settling detection (Stream Watch), and the text reporter (Report).
2. **Grounding** — instruction → cloud vision LLM → pixel point, with strict
   output parsing and fail-closed on unparseable or not-found responses.
3. **Actuator** — click / type / key / scroll / drag / `click_and_type` on the
   live `DISPLAY`, using the existing computer provider.
4. **Wire it into the agent** — semantic click/type targets route through the
   daemon; the brain still receives text only (never a frame).
5. **Continuous frame stream** — keep the ring buffer current so AVCS always
   acts on the current screen.
6. **Remove the reflex / affordance path** from the active perception loop.
7. **Observation feedback** — AVCS's observation goes back to the brain each
   step for replanning.
8. **Honest capability surface** — the operator sees whether AVCS is ready and
   the measured accuracy.

## 11. Definition of done

The design is met only when:

- A GUI action on a described control resolves to real pixels via AVCS.
- AVCS performs a compound `click_and_type` (focus, type, Enter) and reports an
  observation the brain can replan from.
- **No screenshot reaches the brain** — proven by a test, not asserted.
- Reflex and the affordance head are not consulted for perception.
- Accuracy is measured on a real desktop and reported honestly, including
  failure.

Unit tests alone do not satisfy this. A live, isolated desktop task with real
receipts does.
