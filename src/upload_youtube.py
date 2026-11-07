"""
upload_youtube.py
Uploads a finished video to YouTube using the free YouTube Data API v3.

Auth: uses OAuth2 with a stored refresh token (see README "YouTube API
setup" for the one-time process to get client_secret.json and a refresh
token per channel). This script never does the interactive browser login —
that only happens once, locally, when you generate the refresh token.
"""
import os
import pickle
import unicodedata
from pathlib import Path

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


YOUTUBE_TAGS_MAX_CHARS = 500


def _sanitize_youtube_tags(tags: list) -> list:
    """Return unique tags within YouTube's combined 500-character limit."""
    if isinstance(tags, str):
        tags = [tags]

    cleaned = []
    seen = set()
    used_chars = 0
    skipped = 0

    for raw_tag in tags or []:
        if not isinstance(raw_tag, str):
            skipped += 1
            continue

        tag = unicodedata.normalize("NFKC", raw_tag)
        tag = "".join(
            char for char in tag
            if char not in "<>" and not unicodedata.category(char).startswith("C")
        )
        tag = " ".join(tag.replace("#", " ").split()).strip(" ,")
        if not tag:
            skipped += 1
            continue

        key = tag.casefold()
        if key in seen:
            continue

        # YouTube counts separators and quotation marks for tags with spaces.
        tag_cost = len(tag) + (2 if " " in tag else 0)
        separator_cost = 1 if cleaned else 0
        if used_chars + separator_cost + tag_cost > YOUTUBE_TAGS_MAX_CHARS:
            skipped += 1
            continue

        cleaned.append(tag)
        seen.add(key)
        used_chars += separator_cost + tag_cost

    if skipped:
        print(f"WARNING: omitted {skipped} invalid, duplicate, or over-limit YouTube tag(s).")
    print(f"YouTube tags validated: {len(cleaned)} tag(s), {used_chars}/{YOUTUBE_TAGS_MAX_CHARS} characters.")
    return cleaned


def get_authenticated_service(token_path: str, client_secret_path: str = None):
    """
    Loads a saved token (refresh flow, non-interactive — used in CI).
    If no token exists yet, falls back to the interactive browser flow
    (only works locally, not in GitHub Actions).
    """
    creds = None
    token_path = Path(token_path)

    if token_path.exists():
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"WARNING: stored token could not be refreshed ({e}); "
                      f"falling back to interactive login.")
                creds = None
        if not creds:
            if not client_secret_path:
                raise RuntimeError(
                    f"No valid token at {token_path} and no client_secret "
                    "provided for interactive login. Run this script locally "
                    "first to generate the token (see README)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return build("youtube", "v3", credentials=creds)


def verify_token_valid(token_path: str, client_secret_path: str = None) -> None:
    """Fast pre-flight auth check -- call this BEFORE any expensive
    FreeInference/TTS/Pexels work, not just right before the upload call.

    Forces a genuine token refresh (regardless of whether the access
    token looks locally unexpired) -- this is the exact same operation
    that fails with invalid_grant if the refresh token was revoked, and
    unlike calling a YouTube API endpoint, refreshing needs no particular
    API scope, so it can't fail on a scope mismatch the way an unrelated
    read call could.

    Raises RuntimeError with a clear message on any failure."""
    token_path = Path(token_path)
    if not token_path.exists():
        raise RuntimeError(
            f"No token file at {token_path}. Run this script locally "
            f"first to generate one (see README)."
        )
    with open(token_path, "rb") as f:
        creds = pickle.load(f)
    if not creds or not creds.refresh_token:
        raise RuntimeError(
            f"Token at {token_path} has no refresh_token -- regenerate "
            f"it locally (see README)."
        )
    try:
        creds.refresh(Request())
    except Exception as e:
        raise RuntimeError(
            f"Pre-flight auth check failed for token {token_path}: {e}. "
            f"The token is likely revoked/expired -- regenerate it "
            f"locally (see README) and update the corresponding "
            f"GitHub secret."
        )

def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: list,
    category_id: str,
    token_path: str,
    client_secret_path: str = None,
    privacy_status: str = "public",
    is_short: bool = False,
    thumbnail_path: str = None,
    made_for_kids: bool = False,
) -> str:
    """Returns the uploaded video's YouTube ID."""
    youtube = get_authenticated_service(token_path, client_secret_path)

    # YouTube treats videos as Shorts automatically if they're vertical,
    # under 3 minutes, and the title/description mention #Shorts.
    if is_short and "#shorts" not in description.lower():
        description = f"{description}\n\n#Shorts"

    safe_tags = _sanitize_youtube_tags(tags)
    snippet = {
        "title": title[:100],  # YouTube title limit
        "description": description[:5000],
        "categoryId": category_id,
    }
    if safe_tags:
        snippet["tags"] = safe_tags

    body = {
        "snippet": snippet,
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": made_for_kids,
        },
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")

    video_id = response["id"]

    if thumbnail_path:
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
            ).execute()
            print(f"Custom thumbnail set for {video_id}")
        except Exception as e:
            # A thumbnail failure should never take down an otherwise-
            # successful upload. Common cause: the channel isn't phone-
            # verified yet -- custom thumbnails require verification
            # (https://www.youtube.com/verify). The video itself is fine;
            # YouTube just falls back to an auto-picked frame.
            print(f"WARNING: could not set custom thumbnail ({e}). "
                  f"Video uploaded fine, just without a custom thumbnail. "
                  f"If your channel isn't phone-verified, that's the likely cause.")

    return video_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("title")
    parser.add_argument("description")
    parser.add_argument("token_path")
    parser.add_argument("--client_secret", default=None)
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--category_id", default="27")
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()

    video_id = upload_video(
        args.video_path, args.title, args.description, args.tags,
        args.category_id, args.token_path, args.client_secret,
        is_short=args.short,
    )
    print(f"Uploaded: https://youtube.com/watch?v={video_id}")
