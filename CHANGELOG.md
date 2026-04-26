# Changelog

All notable changes to this project should be documented in this file.

## v0.9.1

Fixed
- Corrected backend routing in `/chat/stream` so that requests using `together`, `deepinfra`, `fireworks`, and `baseten` are properly routed to the OpenAI-compatible stream instead of falling back to Hugging Face Inference API.

Added
- Added direct exact-match probing for `fireworks`, `together`, and `baseten` models. This allows users to manually type a model ID (like `accounts/fireworks/models/kimi-k2p6`) in the search bar and use it even if the provider does not list it publicly in their `/models` catalog.

## v0.9

Initial tracked version for LLMFront change history.

Added
- Explicit project versioning starting at `0.9`.
- Provider selection across chat, search, and configuration views.
- Support for additional OpenAI-compatible providers: Together, DeepInfra, Fireworks, and Baseten.
- Model details modal improvements with more metadata and provider-aware API status.
- HF API compatibility checks and fallback logic for certain non-chat models.

Changed
- Provider selection is now persisted across reloads.
- API recommendation logic in the model modal is now more conservative and provider-aware.
- Frontend can recognize provider credentials configured from `.env` without overwriting user inputs.
