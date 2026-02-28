# Kimchi

Kimchi is a **codex-only** local memory/index tool implemented in exactly **3 Python files**:

- `kimchi.py` – CLI, schema, indexing logic
- `kimchi_embed.py` – URL-based embedding client (`POST {inputs, truncate}`)
- `kimchi_query.py` – read-only SQL + semantic retrieval

No ONNX. No Claude Code module.

## Install

```bash
pip install -e .
```

## Embedding endpoint

Default endpoint:

- `https://dev-bge-m3-embedding.sionic.im/embed`

Override with env var:

```bash
export KIMCHI_EMBED_URL="https://dev-bge-m3-embedding.sionic.im/embed"
```

## Usage

Initialize and index Codex sessions from `~/.codex/sessions`:

```bash
kimchi init
```

Quick dry run:

```bash
kimchi init --max-sessions 10
```

Re-index:

```bash
kimchi index
```

Schema overview:

```bash
kimchi search @search
```

Read-only SQL:

```bash
kimchi search "SELECT card_key, text_body FROM pantry_cards LIMIT 5"
```

Semantic search:

```bash
kimchi search --semantic "find tool calls about sqlite" --limit 5
```

Schema similarity search:

```bash
kimchi search --schema-similar "table for tool call logs" --limit 5
```

## Codex skill install (local)

Install this skill directory:

```bash
mkdir -p ~/.codex/skills/kimchi-schema-radar
```

Then place a `SKILL.md` there that runs:

```bash
kimchi search --schema-similar "<user prompt>" --limit 10
```
