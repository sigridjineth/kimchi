# Kimchi

Kimchi is a local Codex memory engine with exactly 3 runtime files:

- `kimchi.py` — CLI, indexing pipeline, skill generation
- `kimchi_embed.py` — HTTP embedding client (`POST { "inputs": [...], "truncate": true }`)
- `kimchi_query.py` — SQL, semantic retrieval, file-scoped history retrieval

## Install

```bash
pip install -e .
```

## Embedding Endpoint

Default:

- `https://dev-bge-m3-embedding.sionic.im/embed`

Override:

```bash
export KIMCHI_EMBED_URL="https://dev-bge-m3-embedding.sionic.im/embed"
```

Optional strict mode (fail fast if embedding API is unavailable):

```bash
export KIMCHI_EMBED_STRICT=1
```

## Quick Start

Initialize DB + index local Codex sessions:

```bash
kimchi init \
  --home ~/.kimchi \
  --cell codex_code \
  --session-root ~/.codex/sessions
```

Re-index everything:

```bash
kimchi index
```

Index one session only:

```bash
kimchi index --session-id "<session-id>"
```

## Search

Schema overview preset:

```bash
kimchi search @search
```

SQL query:

```bash
kimchi search "SELECT card_key, text_body FROM pantry_cards LIMIT 5"
```

Global semantic search:

```bash
kimchi search --semantic "find timeout-related tool calls" --limit 10
```

File-scoped history search (only sessions that touched this file):

```bash
kimchi search --file "src/service/reranker.py" --limit 20
```

File-scoped semantic history search:

```bash
kimchi search --file "src/service/reranker.py" --semantic "retry or timeout handling" --limit 10
```

Schema similarity search:

```bash
kimchi search --schema-similar "table for tool call logs" --limit 5
```

## Generate Skills From Sessions

Single session -> single skill:

```bash
kimchi make-skill \
  --session-id "<session-id>" \
  --name "kimchi-session-radar"
```

Multiple sessions -> multiple named skills:

```bash
kimchi make-skill \
  --session-id "<session-a>" --session-id "<session-b>" \
  --name "qdrant-scala-builder" --name "kimchi-session-radar"
```

File-matched recent sessions -> generated skill names:

```bash
kimchi make-skill \
  --file "qdrant" \
  --top-sessions 2 \
  --name-prefix "qdrant-scala-builder"
```

Destination defaults to `~/.codex/skills`.

Typical output:

```text
• Created 2 new skill(s) based on those sessions
  and validated them.

  - qdrant-scala-builder/SKILL.md
  - kimchi-session-radar/SKILL.md

  Validation:

  - quick_validate.py qdrant-scala-builder →
    Skill is valid!
  - quick_validate.py kimchi-session-radar →
    Skill is valid!

  You can now invoke them with:

  - $qdrant-scala-builder
  - $kimchi-session-radar
```

## Session-End Auto Index Hook

This repository includes:

- `.omx/hooks/kimchi-session-end-embed.mjs`

Behavior:

- Trigger: `session-end`
- Action: background run of `kimchi index --session-id <ended-session-id>`

Enable hook plugins:

```bash
export OMX_HOOK_PLUGINS=1
```

Recommended env:

```bash
export KIMCHI_HOME="$HOME/.kimchi"
export KIMCHI_SESSION_ROOT="$HOME/.codex/sessions"
```

Check hook status:

```bash
omx hooks status
omx hooks validate
```

## CLI Summary

```bash
kimchi init
kimchi index
kimchi search
kimchi make-skill
```
