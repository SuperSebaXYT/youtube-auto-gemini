"""Read all uploads and metadata from the authenticated YouTube channel."""
from __future__ import annotations

import pickle
from pathlib import Path

from google.auth.transport.requests import Request
from googleapiclient.discovery import build


def fetch_channel_videos(token_path: str) -> list[dict]:
    path = Path(token_path)
    if not path.exists():
        raise FileNotFoundError(f"YouTube read-only token not found: {path}")
    with path.open("rb") as handle:
        credentials = pickle.load(handle)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        with path.open("wb") as handle:
            pickle.dump(credentials, handle)
    if not credentials.valid:
        raise RuntimeError("YouTube read-only token is invalid or expired.")
    youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)

    channel_response = youtube.channels().list(
        part="contentDetails", mine=True, maxResults=1
    ).execute()
    channels = channel_response.get("items", [])
    if not channels:
        raise RuntimeError("OAuth account has no accessible YouTube channel.")
    uploads_id = channels[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    video_ids = []
    page_token = None
    while True:
        kwargs = {"part": "contentDetails,snippet", "playlistId": uploads_id, "maxResults": 50}
        if page_token:
            kwargs["pageToken"] = page_token
        response = youtube.playlistItems().list(**kwargs).execute()
        for item in response.get("items", []):
            video_id = item.get("contentDetails", {}).get("videoId")
            if video_id:
                video_ids.append(video_id)
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    results = []
    for offset in range(0, len(video_ids), 50):
        batch = video_ids[offset:offset + 50]
        response = youtube.videos().list(
            part="snippet,contentDetails,status", id=",".join(batch), maxResults=50
        ).execute()
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            status = item.get("status", {})
            results.append({
                "video_id": item.get("id"),
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "published_at": snippet.get("publishedAt"),
                "duration": item.get("contentDetails", {}).get("duration"),
                "tags": snippet.get("tags", []),
                "category_id": snippet.get("categoryId"),
                "privacy_status": status.get("privacyStatus"),
                "made_for_kids": status.get("madeForKids"),
            })
    return results
