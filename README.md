# AutoTube — 3-channel autonomous YouTube pipeline

Generates and uploads 1 long-form video + 5 Shorts per day, per channel,
across 3 channels, fully free, running on GitHub Actions (no server needed).

## What runs where

- **Code**: lives in this repo, written and maintained here.
- **Execution**: GitHub Actions, on a free-tier schedule. This is what makes
  it "autonomous" — once set up, it runs without you opening a laptop.
- **You**: only needed for the one-time setup below (~1-2 hours total across
  3 channels), and periodically topping up the topic lists.

## PART 1 — One-time setup (do this once per channel, 3x total)

### 1. Create the YouTube channel
Go to youtube.com → click your profile icon → "Create a channel". Do this
under the same Google account, or separate accounts if you want fully
independent channels (separate accounts avoids any single strike affecting
all 3 — up to you).

### 2. Get a FreeInference API key (text + visual AI)
1. Go to https://freeinference.org/ and create/sign in to your account.
2. Create an API key in your account dashboard.
3. Confirm that your account exposes `deepseek-v4-flash` and `qwen3.6-35b` at https://freeinference.org/v1/models.

Add the key to GitHub Actions as `FREEINFERENCE_API_KEY`. The pipeline uses DeepSeek V4 Flash for text generation and Qwen3.6-35B for Visual QA. The `bge-m3` model is embeddings-only and is not used for chat or image verification.

### 3. Get a free Pexels API key (for stock footage)
1. Go to https://www.pexels.com/api/
2. Sign up, confirm email, copy your API key

One key works for all 3 channels (used as `PEXELS_API_KEY`).

### 4. Get YouTube API credentials (Google Cloud Console)
This is the fiddliest part — go slowly, it's one-time.

1. Go to https://console.cloud.google.com/
2. Create a new project (any name, e.g. "autotube")
3. In the search bar, find **"YouTube Data API v3"** → click **Enable**
4. Go to **APIs & Services → OAuth consent screen**
   - User type: External
   - Fill app name (e.g. "AutoTube"), your email for support/dev contact
   - Add your own Google account email as a **Test user**
   - Save
5. Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**
   - Application type: **Desktop app**
   - Name it anything
   - Click Create, then **Download JSON** — save it as `client_secret.json`

You only need to do steps 2-5 **once total** (one Google Cloud project can
issue credentials for all 3 channels — you'll just authorize it 3 times,
once per channel, in the next step).

### 5. Generate a refresh token per channel (run this locally, once per channel)
This is the only step that needs your local machine and a browser — it's
how you grant this pipeline permission to upload to each channel.

```bash
git clone <your-repo-url>
cd autotube
pip install -r requirements.txt

# Run once per channel, logging into the matching Google/YouTube account
# when the browser window opens:
python -c "
from src.upload_youtube import get_authenticated_service
get_authenticated_service('tokens/mind_bites_token.pickle', 'client_secret.json')
"
# repeat for then_and_now_token.pickle and money_simple_token.pickle,
# logging into the correct channel's account each time
```

This creates `tokens/<channel>_token.pickle` — the credential GitHub
Actions will use to upload on your behalf, without ever needing your
password again.

### 6. Add secrets to GitHub
In your repo: **Settings → Secrets and variables → Actions → New repository secret**

| Secret name | Value |
|---|---|
| `FREEINFERENCE_API_KEY` | from step 2 |
| `PEXELS_API_KEY` | from step 3 |
| `YOUTUBE_TOKEN_MIND_BITES` | `base64 -i tokens/mind_bites_token.pickle` output |
| `YOUTUBE_TOKEN_THEN_AND_NOW` | `base64 -i tokens/then_and_now_token.pickle` output |
| `YOUTUBE_TOKEN_MONEY_SIMPLE` | `base64 -i tokens/money_simple_token.pickle` output |

To get the base64 value:
```bash
base64 -i tokens/mind_bites_token.pickle | tr -d '\n'
```
(On Linux, `base64 -w0` does the same thing without line breaks.)

**Never commit `client_secret.json` or the `.pickle` token files to the
repo** — `.gitignore` below already excludes them, but double-check.

### 7. Push to GitHub and enable Actions
```bash
git init
git add .
git commit -m "Initial pipeline"
git remote add origin <your-repo-url>
git push -u origin main
```
Then go to the **Actions** tab in your repo and enable workflows if prompted.

## PART 2 — Testing before going live

**Always test with `--dry-run` first** so you review output before it goes
public:

```bash
python src/main.py mind_bites --dry-run
python src/main.py mind_bites --short --dry-run
```

This builds the video and saves it to `output/` instead of uploading —
check the video actually looks right, audio is synced, captions read
correctly, before trusting the automation.

You can also trigger a real single test run from GitHub: go to
**Actions → Daily Upload Pipeline → Run workflow**, fill in a channel name,
and toggle `dry_run` on or off.

## PART 3 — How the schedule works

`src/scheduler.py` runs every hour (triggered by GitHub Actions cron). It
checks each channel's YAML config for `upload_time_utc` (1 long-form) and
`shorts_upload_times_utc` (5 times/day) and only generates+uploads what's
due at the current hour. Adjust those times in the channel YAML files to
change the posting schedule — times are in UTC.

## Keeping topics fresh

Each channel's topic list lives in `channels/<channel>_topics.txt` — one
topic per line, ~20 to start. The pipeline picks a random unused one each
run and won't repeat until the list is exhausted (then it loops). **Add
20-30 more topics every few weeks** so content doesn't start repeating —
just append new lines to the relevant `.txt` file and commit.

## Costs — what's actually free vs. what to watch

| Component | Free tier | Watch out for |
|---|---|---|
| FreeInference API | Account-specific daily quota and concurrency limits | Check your dashboard and https://doc.freeinference.org/models |
| Pexels API | Free, rate-limited | 200 requests/hour on free tier — plenty for this volume |
| Edge-TTS | Free, unofficial API | Not officially supported by Microsoft; could break without notice |
| YouTube Data API | 10,000 units/day free | Each upload costs ~1600 units → ~6 uploads/day is close to the ceiling. 3 channels × 6 = 18 uploads would need 3 separate Cloud projects/quotas, OR you reduce volume. **See note below.** |
| GitHub Actions | 2,000 free minutes/month (public repos: unlimited) | Make the repo public to avoid any minute limits, or watch usage if private |

### Important: YouTube API quota with 3 channels
The 10,000 units/day quota is **per Google Cloud project**, not per
channel. If all 3 channels' tokens come from the same Cloud project, you
have one shared 10,000-unit pool — 18 uploads/day (≈28,800 units) would
exceed it.

**Fix**: create 3 separate Google Cloud projects (one per channel) in step
4 above, each enabled for YouTube Data API v3, each with its own OAuth
client. Then each channel gets its own free 10,000-unit daily quota. This
adds ~10 minutes of repeated clicking per extra channel but costs nothing.

## Content policy note

YouTube increasingly demonetizes or suppresses "reused/low-effort"
AI-generated content. This pipeline gives you a working base, but for real
channel growth you'll likely want to:
- Review/edit scripts before they go fully public (start with `--dry-run`
  or `privacy: unlisted` for the first couple weeks)
- Add a distinct visual identity (consistent intro, channel-branded
  thumbnail template) rather than relying purely on stock footage
- For **Money Simple** specifically: YouTube scrutinizes finance content
  (YMYL policy) more heavily — keep scripts educational, never
  prescriptive ("you should buy X")
