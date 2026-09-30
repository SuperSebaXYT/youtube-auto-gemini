from src.trend_scout import _fetch_reddit_trending, _fetch_youtube_trending


def collect(config: dict) -> dict:
    niche = config.get("niche") or "Animals & wildlife facts"

    # Keep trend discovery strictly within the animal/wildlife niche.
    youtube = _fetch_youtube_trending(niche, 15)

    subreddit = config.get("trend_subreddit") or "animals"
    reddit = _fetch_reddit_trending(subreddit, 12)

    return {
        "youtube": youtube,
        "reddit": reddit,
    }