from __future__ import annotations

import os

from huggingface_hub import HfApi


class HuggingFaceAuthError(RuntimeError):
    """Raised when Hugging Face authentication or repo access is invalid."""


def validate_hf_access(
    *,
    repo_id: str | None = None,
    repo_type: str = "dataset",
    token: str | None = None,
) -> HfApi:
    resolved_token = token or os.getenv("HF_TOKEN")
    if not resolved_token:
        raise HuggingFaceAuthError(
            "Hugging Face authentication failed: HF_TOKEN is missing."
        )

    api = HfApi(token=resolved_token)

    try:
        api.whoami(token=resolved_token)
    except Exception as exc:
        raise HuggingFaceAuthError(
            "Hugging Face authentication failed: HF_TOKEN is invalid, expired, or revoked."
        ) from exc

    if repo_id is not None:
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
        except Exception as exc:
            raise HuggingFaceAuthError(
                f"Hugging Face authentication failed: HF_TOKEN does not have access to {repo_type} repo '{repo_id}'."
            ) from exc

    return api
