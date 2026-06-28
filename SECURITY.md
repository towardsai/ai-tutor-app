# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately through GitHub's [**Report a vulnerability**](https://github.com/towardsai/ai-tutor-app/security/advisories/new) button (the repo's Security tab -> Advisories). That opens a private advisory visible only to the maintainers. We aim to acknowledge reports within a few business days.

If you would rather start over email, contact a maintainer (see the [README](./README.md)) and we will move the discussion into a private advisory.

## Scope

This repository holds the **application code** for the AI Tutor. The course corpus and embeddings live in **private** Hugging Face datasets gated by `HF_TOKEN`; they are not in this repo.

## Never commit

- **API keys or tokens.** Use `.env` (gitignored). Secret scanning and push protection are enabled on this repo, but treat them as a backstop, not a reason to be careless.
- **Private or user data.** Anything containing real user text or other private data must stay out of git. Pushes to `main` are deployed to a public Hugging Face Space, so a committed data file becomes world-readable on the next deploy. See `AGENTS.md` -> "Gotchas".

If you discover a committed secret or private-data file, treat it as a security incident: report it privately and rotate the affected credential immediately.
