<!-- forkable-card:start (added by apply_cards.py; re-run it to update) -->
![Playwright Self-Healer: self-healing Playwright tests from plain-English specs, repairing broken locators offline](.github/assets/forkable-card.png)
<!-- forkable-card:end -->

# ForkablePlaywrightSelfHealer

[![CI](https://github.com/suryakulshreshtha/ForkablePlaywrightSelfHealer/actions/workflows/ci.yml/badge.svg)](https://github.com/suryakulshreshtha/ForkablePlaywrightSelfHealer/actions/workflows/ci.yml)

Skunkworks projects exploring what local, offline AI agents can actually do —
no cloud APIs, no telemetry, no network.

## Projects

### [PlaywrightSelfHealer](PlaywrightSelfHealer/) — self-healing Playwright tests, offline

Turns plain English into Playwright tests, runs them, repairs the selectors your
app broke overnight, explains failures against a local knowledge base, and emits
standalone pytest code. Healing is heuristic-first; a local Ollama model makes
planning more flexible and diagnoses more readable, but nothing here requires it.

- **Stack:** Python 3.11+, Playwright, Ollama (`qwen2.5-coder`, `nomic-embed-text`), hybrid BM25 + dense RAG, stdlib everywhere else — one runtime dependency
- **Highlight:** the same spec passes before and after a UI refactor that deletes every `data-testid` and renames every id — the agent re-binds and remembers
- **Offline:** the socket layer is patched so non-loopback traffic and DNS both raise before a packet leaves the process
- **API + UI in one plan:** API steps run through the browser's cookie jar, so a UI login authenticates the API call that follows
- **Verified:** 125 tests, including real Chromium runs; two fake servers (a mini headless browser and an Ollama stand-in) keep coverage honest with no network and no GPU
- **Not an autonomous agent:** no goal decomposition, no replanning loop. It executes a plan and adapts its locators — see the README for the honest breakdown

```bash
cd PlaywrightSelfHealer
pip install -e ".[dev,visual]"
python -m playwright install chromium
forkable doctor
forkable demo
```

See [PlaywrightSelfHealer/README.md](PlaywrightSelfHealer/README.md) for the full write-up,
[RESULTS.md](PlaywrightSelfHealer/RESULTS.md) for the verification table, and
[PUBLISHING.md](PUBLISHING.md) for repository details and the macOS publish process.

## Sibling repository

Companion to [ForkedUpAIExperiments](https://github.com/suryakulshreshtha/ForkedUpAIExperiments)
(local PDF RAG in Python, autonomous ADO code review agent in Go). Same
philosophy — enterprise-grade agents on consumer hardware, nothing leaving the
machine — and the same local model stack, so the two can share an Ollama install.

## Licence

MIT.
