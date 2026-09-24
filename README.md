# p2i-core

The Python engine for P2I: PyTorch tracing, observed Model IR, editable ArchitectureIR, module adapters, Harness, validation, build/retrace, skills and evaluation.

## Boundary

This repository will own the existing `p2i` Python import and its JSON schemas. Core must run locally without a browser, FastAPI, CLI or AI provider. It must not import the SDK, GUI, CLI or AI plugin.

## Migration status

**Repository initialized; implementation has not moved yet.** The tested package and examples remain at [PtwoI/p2i](https://github.com/PtwoI/p2i). Continue installing and running that repository until the extraction passes the original Python tests, IR compatibility tests and package smoke tests.

## Related components

- [p2i-sdk](https://github.com/PtwoI/p2i-sdk): typed client and tool protocol
- [p2i-gui](https://github.com/PtwoI/p2i-gui): local web explorer
- [p2i-cli](https://github.com/PtwoI/p2i-cli): terminal commands
- [p2i-ai-plugin](https://github.com/PtwoI/p2i-ai-plugin): optional external-agent adapters

MIT licensed.
