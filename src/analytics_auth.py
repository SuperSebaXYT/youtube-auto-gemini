"""Strict OAuth authentication for the YouTube Analytics API.

The Analytics credential is separate from the YouTube upload credential.
A revoked refresh token cannot be repaired by retrying or by opening a
browser inside GitHub Actions, so authentication errors are surfaced
immediately and clearly.
"""
import pickle
from pathlib import Path

from google.auth.transport.requests import Request
from googleapiclient.discovery import build

ANALYTICS_SCOPES = frozenset({
    "https://www.googleapis.com/auth/yt-analytics.readonly"
})


def load_analytics_credentials(token_path: str):
    path = Path(token_path)
    if not path.is_file():
        raise RuntimeError(
            f"YouTube Analytics token file is missing: {path}. "
            "Restore YOUTUBE_ANALYTICS_TOKEN_COSMIC_CURIOUS in GitHub Secrets."
        )

    try:
        with path.open("rb") as token_file:
            creds = pickle.load(token_file)
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode the Analytics token at {path}. "
            "The GitHub secret must be the Base64 encoding of the original "
            "OAuth credentials pickle file."
        ) from exc

    if not hasattr(creds, "refresh_token") or not hasattr(creds, "scopes"):
        raise RuntimeError(
            f"The file at {path} is not a Google OAuth credentials object."
        )

    if not creds.refresh_token:
        raise RuntimeError(
            f"The Analytics token at {path} has no refresh token. "
            "It is not usable for unattended GitHub Actions runs."
        )

    scopes = set(creds.scopes or [])
    if not ANALYTICS_SCOPES.issubset(scopes):
        raise RuntimeError(
            "The restored OAuth token does not contain the required "
            "yt-analytics.readonly scope. A YouTube upload/read-only token "
            "cannot be used for Analytics."
        )

    return path, creds


def get_analytics_service(token_path: str, client_secret_path: str = None):
    """Refresh the stored Analytics credential and build the API client.

    client_secret_path remains accepted for backwards compatibility, but
    interactive browser login is intentionally not attempted in CI.
    """
    path, creds = load_analytics_credentials(token_path)

    if creds.expired or not creds.valid:
        try:
            creds.refresh(Request())
        except Exception as exc:
            raise RuntimeError(
                "Google rejected the YouTube Analytics refresh token "
                f"({type(exc).__name__}: {exc}). The restored pickle was "
                "read successfully and contains the Analytics scope, but "
                "Google did not accept its refresh token. Check that the "
                "GitHub secret contains the exact newly generated Analytics "
                "pickle and that it was created with the same OAuth client."
            ) from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as token_file:
            pickle.dump(creds, token_file)
        temporary_path.replace(path)

    return build(
        "youtubeAnalytics",
        "v2",
        credentials=creds,
        cache_discovery=False,
    )
