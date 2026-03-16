# RAEON Architecture Spec v1

## Purpose

RAEON is not a chatbot-first system. It is an evolving identity architecture where
memory, traits, and personality emerge over time from accumulated experience.

This document turns the current RAEON vision into a practical technical blueprint
that can guide implementation while preserving the original philosophy:

- Nothing should be fully fixed.
- Memory is the base of identity.
- Personality should emerge, not be hardcoded.
- Perception, memory, traits, and behavior should remain loosely coupled.
- The architecture must allow open-ended evolution.

## System Positioning

RAEON should be treated as the full mind architecture.

theSpaceDB should be treated as the memory substrate inside that architecture.

In simple terms:

- `theSpaceDB` stores evolving memory geometry.
- `RAEON` interprets that geometry into traits, personality, and decisions.

## Core Flow

The main loop for a single experience is:

1. Perception receives an input event.
2. The event is normalized into a memory candidate.
3. A memory block is created and stored.
4. The new block is linked to prior blocks chronologically and semantically.
5. Repeated patterns are analyzed for trait signals.
6. Personality state is updated from trait shifts and reinforcement.
7. The decision engine produces an action, response, or no-op.
8. Outcome and feedback are stored as new experience.

This creates a recursive identity loop:

`Perception -> Memory -> Traits -> Personality -> Decision -> Outcome -> Memory`

## Layers

### 1. Perception Layer

Responsible for turning raw input into structured experience packets.

Inputs may eventually include:

- text
- image frames
- video segments
- audio
- touch or sensor events
- internal signals
- action outcomes

This layer should not decide personality. It only produces interpretable events.

Suggested output model:

- raw payload
- source type
- timestamp
- environment metadata
- preliminary salience
- optional preprocessing outputs

### 2. Memory Layer

Responsible for persistent storage of experiences as memory blocks.

This is the foundational identity layer. Every experience becomes a memory block.

Memory responsibilities:

- store blocks durably
- preserve chronology
- preserve semantic relationships
- support multimodal payloads
- support reinforcement and decay
- support retrieval under time constraints

This layer maps well onto theSpaceDB.

### 3. Relationship Layer

Responsible for changing memory geometry over time.

This is where the original SpaceDB philosophy is strongest:

- blocks begin mostly disconnected
- co-occurrence pulls them closer
- inactivity or conflicting context may push them apart
- neighborhoods become clusters
- mature clusters become persistent contexts

Important dynamic rules:

- closeness should increase with repeated co-activation
- separation should increase with disuse, contradiction, or context drift
- rate of separation may scale with incoming experience rate
- drift can reshape memory even without external input

### 4. Trait Layer

Responsible for discovering repeated behavioral/cognitive patterns across memory.

Traits are not manually assigned labels in the core engine. They should emerge from:

- repeated decisions
- repeated emotional tags
- repeated outcomes
- reinforcement patterns
- conflict resolution patterns
- preference stability over time

Traits should be represented as evolving weighted signals, not booleans.

Examples:

- curiosity
- caution
- aggression
- attachment
- openness
- persistence
- avoidance

### 5. Personality Layer

Responsible for maintaining long-running identity state.

Personality is not one static profile. It is:

- a weighted state derived from traits
- a set of active contexts
- a record of long-term tendencies
- a selector for how decisions should be biased

In the original SpaceDB framing, clusters can become personalities.

In RAEON terms:

- a cluster is a stable memory neighborhood
- a personality is a cluster or context that has reached enough weight,
  recurrence, and behavioral relevance to influence output

This promotion threshold should remain adaptive, not fixed forever.

### 6. Decision Layer

Responsible for turning current state into output.

Decision inputs:

- current input event
- active personality context
- closest relevant memories
- emotional/importance weighting
- current time budget for output
- recent outcomes and current goals

Decision output may be:

- response text
- action
- plan
- internal state change
- deliberate silence

### 7. Drift / Thought Layer

Responsible for self-evolution when no external input is present.

This is the "thinking" mechanism.

When idle, the system should:

- pick a random block, cluster, or personality
- perform controlled random walks
- reinforce some weak edges
- decay some stale edges
- explore cross-cluster bridges
- create speculative links with very low confidence

This should be slow, bounded, and auditable.

Goal:

- preserve movement without destabilizing identity
- allow novelty and emergent cross-domain association

## Canonical Data Models

### MemoryBlock

This should be the canonical experience record.

Suggested fields:

```json
{
  "id": "uuid",
  "timestamp_utc": "2026-03-11T12:00:00Z",
  "source_type": "text|vision|audio|touch|internal|action|feedback",
  "raw_input": {},
  "normalized_content": "cleaned meaning representation",
  "embedding": [],
  "emotion": {
    "valence": 0.0,
    "arousal": 0.0,
    "tags": []
  },
  "importance": 0.0,
  "novelty": 0.0,
  "confidence": 0.0,
  "linked_previous": [],
  "semantic_neighbors": [],
  "cluster_ids": [],
  "active_traits": {},
  "personality_context": [],
  "action_taken": null,
  "outcome": null,
  "feedback": null,
  "metadata": {}
}
```

Notes:

- `raw_input` preserves the original observation.
- `normalized_content` is what downstream systems reason over.
- `embedding` can be stored externally if size becomes a concern.
- `linked_previous` is chronological memory chaining.
- `semantic_neighbors` is geometric or learned relation.
- `active_traits` is a snapshot at write time, not the global truth.

### TraitSignal

Suggested structure:

```json
{
  "name": "curiosity",
  "weight": 0.42,
  "confidence": 0.78,
  "sources": ["memory-pattern", "outcome-feedback"],
  "last_updated_utc": "2026-03-11T12:00:00Z"
}
```

Traits should be fluid and recalculable.

### PersonalityState

Suggested structure:

```json
{
  "id": "personality-cluster-id",
  "name": "explorer",
  "status": "active|latent|emergent|deprecated",
  "dominant_traits": {
    "curiosity": 0.82,
    "risk_tolerance": 0.41
  },
  "source_clusters": [],
  "activation_score": 0.0,
  "stability_score": 0.0,
  "min_blocks_required": 0,
  "last_activated_utc": "2026-03-11T12:00:00Z",
  "metadata": {}
}
```

Important:

- `min_blocks_required` should be adaptive.
- `activation_score` decides how likely this personality is to influence output now.
- `stability_score` helps distinguish noise from identity.

## Retrieval Model

Output should be time-aware.

Original principle:

- maximum retrieval distance should be directly proportional to allowed output time

Practical interpretation:

- short output window -> small search radius, fewer memories, more reflexive output
- long output window -> wider radius, more memories, more deliberation

Suggested retrieval stages:

1. Resolve active personality context.
2. Determine output time budget.
3. Convert time budget into retrieval radius and candidate count.
4. Retrieve nearest memories within that radius.
5. Re-rank by importance, emotion, novelty, and outcome relevance.
6. Feed ranked memories into decision generation.

Example control mapping:

- `response_window_ms`
- `max_distance`
- `candidate_limit`
- `cross_cluster_allowance`

## Dynamic Rules

These are the most important behavioral rules to preserve from the original idea.

### Rule 1: Initial Isolation

At early stages, memory blocks should not assume rich structure. Most closeness must
be earned through repeated experience.

### Rule 2: Reinforcement Through Co-Experience

Blocks that repeatedly activate together should move closer.

### Rule 3: Separation Through Growth

The rate at which two blocks drift apart may depend on the rate of new incoming
experience. A rapidly learning system should restructure faster than a slow one.

### Rule 4: Cluster Emergence

Clusters should emerge from geometry and interaction patterns, not from fixed labels.

### Rule 5: Personality Promotion

A cluster becomes personality-relevant only when it crosses an adaptive threshold
of size, recurrence, stability, and behavioral impact.

### Rule 6: Idle Thinking

In the absence of input, the system should continue evolving via constrained drift,
random walks, and low-rate cross-context exploration.

### Rule 7: Feedback-Driven Identity

Outcomes of decisions must go back into memory and influence future traits and
personality weights.

## Proposed Module Boundaries

Suggested implementation modules for RAEON:

- `Perception`
  - input normalization
  - multimodal adapters

- `Memory`
  - memory block schema
  - storage
  - chaining
  - retrieval

- `Linking`
  - semantic edges
  - chronological edges
  - reinforcement/decay

- `Traits`
  - pattern extraction
  - trait scoring

- `Personality`
  - cluster promotion
  - active context selection
  - long-term state persistence

- `Decision`
  - output planning
  - action selection
  - response shaping

- `Drift`
  - idle cognition
  - random walk logic
  - background evolution

- `Feedback`
  - outcome capture
  - reinforcement from results

## Relationship to NextMind

The current `NextMind` codebase appears to be an application/product layer.
It should not become the place where the full evolving identity model is tightly
mixed into UI and request handling.

Recommended direction:

- keep `NextMind` as an application shell or product host
- keep `theSpaceDB` as the memory substrate
- introduce `RAEON` as a distinct domain layer or service boundary

That keeps the system modular and easier to test.

## Immediate Build Plan

These are the best next concrete milestones.

### Phase 1: Canonical Memory

Build:

- `MemoryBlock` schema
- persistent storage contract
- chronological linking
- semantic linking hooks

Success condition:

- one experience can be stored, linked, and retrieved consistently

### Phase 2: Trait Prototype

Build:

- trait extraction from repeated memory patterns
- weighted trait state per time window

Success condition:

- repeated experience changes trait weights in a visible way

### Phase 3: Personality State

Build:

- cluster-to-personality promotion
- adaptive threshold logic
- active personality selection

Success condition:

- stable memory neighborhoods begin influencing decisions

### Phase 4: Decision Engine v1

Build:

- personality-conditioned retrieval
- time-window-based recall
- simple response/action generation

Success condition:

- different active personalities produce measurably different outputs

### Phase 5: Drift / Thought Engine

Build:

- idle random walk
- low-rate reinforcement/decay
- cross-cluster bridge formation

Success condition:

- the system evolves even without input, but within safe and explainable bounds

## Non-Goals for v1

To keep the first implementation grounded, avoid trying to solve all of this at once.

Do not require in v1:

- full embodiment
- unrestricted self-modification
- perfect human cognition simulation
- heavy multimodal pipelines from day one
- opaque black-box personality changes

## Design Constraints

Every major layer should be:

- independently testable
- serializable
- inspectable
- replayable from stored history
- replaceable without collapsing the whole architecture

This is important because identity without observability becomes impossible to debug.

## Suggested v1 Technical Direction

If implementation starts soon, a clean first cut would be:

- `theSpaceDB` for memory storage and geometric linking
- a `RAEON.Core` domain library for traits, personality, and decision logic
- explicit DTOs for `MemoryBlock`, `TraitSignal`, and `PersonalityState`
- background drift worker separated from request/response APIs
- deterministic test fixtures for replaying identity evolution over time

## Open Questions

These decisions should be answered before heavy implementation:

1. Should traits be fully emergent, or should v1 allow a seeded starter set?
2. How much emotional modeling belongs in the first version?
3. Should personality activation be singular or mixed across multiple clusters?
4. What is the first real-world input modality: text, room events, sensors, or all?
5. How aggressive should idle drift be before it risks destabilizing memory?

## Summary

RAEON should be built as an evolving identity architecture on top of a dynamic
memory substrate.

The essential idea is:

- memory creates structure
- repeated structure creates traits
- stabilized traits create personality
- personality shapes decisions
- decisions create new memory

That loop is the mind.
