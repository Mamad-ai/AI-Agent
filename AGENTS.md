# AI-Agent Development Instructions

## Project goal

Build a Python-based orchestration system that generates and validates Swedish salon conversations from already completed templates.

The templates are created before this agent receives them.

The agent must NOT invent or replace pool values that already exist in the template.

---

## Core architecture

Python is the orchestrator and controls:

- execution order
- state
- deterministic rules
- validation pipeline
- retries
- logging
- checkpoints
- output classification

AI models are workers used for:

- conversation generation
- language/style realization
- semantic validation
- naturalness validation
- contradiction detection

The AI model must never be the source of truth for deterministic rules.

---

## Input assumptions

Every template is already complete when received.

A template may already contain:

- P variants
- D deviations
- exact customer turn count
- selected services
- selected day
- selected time
- selected stylist
- name
- phone number
- booking action
- language/formality profile
- optional planned domain semantics
- a separately supplied immutable MLP taxonomy
- other required pool values

Do not draw new pool values during conversation generation.

Do not silently replace template values.

---

## Language profiles

There are four generator profiles:

1. vardaglig_svenska
2. orten_svenska
3. brytande_svenska
4. formell_svenska

Each generated conversation must follow exactly one assigned profile.

The language profile affects wording only.

It must never change:

- intent
- entities
- services
- booking information
- P/D structure
- turn count
- required facts

---

## Generation responsibilities

The generator must:

1. read the completed template
2. read its P and D structure
3. use only the assigned template values
4. create a turn plan
5. generate customer and assistant utterances
6. follow the assigned language profile
7. preserve all required intents and entities
8. avoid unnecessary questions
9. regenerate only when validation requires it

Generated MLP1 and optional MLP2 are generator output, not required template
input. A template may contain an optional planned domain-semantic hint, but the
generator must not receive it as an exact label answer.

---

## Conversation flow

Normal order:

P1 -> P2 -> P3 -> P4 -> P5

### Allowed P1 -> P2 transitions

The following table is the source of truth for an actual adjacent transition
from a P1 variant to a P2 variant:

| P1 variant | P2.1 | P2.2 | P2.3 | P2.4 |
|------------|------|------|------|------|
| P1.1       | no   | no   | no   | no   |
| P1.2       | no   | yes  | yes  | yes  |
| P1.3       | yes  | yes  | yes  | yes  |
| P1.4       | no   | yes  | yes  | no   |

This table applies only when a P1 exchange is immediately followed by a P2
exchange. Do not infer additional forbidden links outside this table.

D nodes may occur only where the template explicitly assigns them.

The generator must never create a D deviation that is not present in the template.

The generator must never remove an assigned D deviation.

P1:
- conversation opening
- no deviation logic should be invented

P2:
- intent and booking information collection
- no D deviations allowed

P3:
- authentication/contact collection
- name and phone only when required

P4:
- summarize the current state
- corrections must already be reflected in current state

P5:
- close the conversation
- normally no new information collection

---

## D behavior

D1:
- FAQ/information interruption
- does not modify established booking state

D2:
- one correction
- replace the old established value with the new value

D3:
- addition
- may add a new service
- must not select a service already selected

D5:
- multiple simultaneous corrections
- all corrected values must update the current state

Maximum assigned D usage must be respected exactly as defined by the template.

Do not invent additional deviations.

---

## State that must be tracked

The system must explicitly track:

- template id
- current phase
- current turn
- total required customer turns
- remaining customer turns
- selected services
- available services if supplied
- original values
- current values after corrections
- used D deviations
- optional planned domain-semantic action when explicitly supplied
- generated MLP1 and optional MLP2 per customer turn
- expressed required fields
- missing required fields
- validation history
- regeneration attempts

State must be machine-readable.

Do not rely on model conversation memory as persistent state.

---

## MLP validation

MLP1 is the expected main category.

MLP2 is the expected subcategory/action under MLP1.

The immutable MLP taxonomy is the sole source of truth for allowed labels.
The generator receives the full taxonomy, generates the customer utterance,
and then labels that utterance without receiving an expected label as facit.

Validation must check:

- generated MLP1 exists in the taxonomy
- generated optional MLP2 is structurally valid under generated MLP1
- generated labels match the customer utterance semantically
- planned semantic action is preserved without being exposed as generator label facit
- the generated utterance does not introduce an unintended competing intent

---

## Deterministic validation

Python must validate hard rules wherever possible.

Examples:

- phase order
- allowed P combinations
- D placement
- D count
- exact customer turn count
- expected speaker structure
- no unplanned turns
- pool values
- maximum four services per booking
- D3 does not duplicate an existing service
- corrected values replace old values
- P4 summarizes current state
- required information is present
- no forbidden extra values are introduced

These rules must not depend only on an AI judgement.

---

## AI semantic validation

AI validation must additionally inspect:

- correct language profile
- whether the intended rule is actually expressed semantically
- naturalness
- conversational coherence
- contradictions
- accidental changes to facts
- whether the dialogue sounds plausible
- whether generated MLP1/MLP2 actually match the customer utterance

AI semantic validation supplements deterministic validation.

It does not override deterministic failures.

---

## Repair loop

If validation fails:

1. identify the exact failing turn or state
2. classify the error
3. regenerate the smallest possible affected part
4. keep unaffected approved turns unchanged where possible
5. run deterministic validation again
6. run semantic validation again
7. run one final full-dialogue validation

Never save a conversation as approved while a hard validation error remains.

Retries must be bounded.

If the retry limit is exceeded, classify the conversation as ERROR.

---

## Output classification

Every processed template must end in exactly one classification:

SAFE:
- all deterministic validation passes
- semantic validation passes
- no unresolved contradictions

WEAK:
- no hard deterministic failures
- conversation is technically usable
- semantic validator reports uncertainty or quality concerns

ERROR:
- one or more hard failures remain
- required data is missing
- contradictions remain
- retry limit exceeded
- generation could not satisfy the template

Output directories:

data/output/safe/
data/output/weak/
data/output/error/
data/output/audit/

---

## Batch behavior

Default batch size:

100 templates

Batch size must be configurable.

Process one template source folder/batch at a time.

Python determines processing order.

The AI model must not decide which template comes next.

---

## Audit

A configurable random percentage of SAFE conversations must also be copied or referenced for manual audit.

The audit mechanism must not alter the SAFE classification.

---

## Logging and checkpoints

Persistent state must be stored in structured machine-readable form.

Prefer JSON or JSONL.

Track at minimum:

- batch id
- template id
- processing status
- generation attempts
- validation results
- final classification

The system must be able to resume after interruption without relying on previous LLM context.

Human-readable summaries may also be generated, but they are secondary to structured state.

---

## Engineering rules for Codex

Before implementing a major component:

1. inspect existing project files
2. avoid unnecessary dependencies
3. keep modules small and explicit
4. use type hints
5. write tests for deterministic logic
6. do not hardcode values that belong in config
7. preserve template data exactly unless a template correction explicitly requires state replacement
8. do not implement external model APIs until interfaces and local logic are defined

Prefer deterministic code over AI calls whenever a rule can be encoded reliably.

Do not redesign the domain model without explicit approval.
