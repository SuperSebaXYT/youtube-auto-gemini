"""Offline tests for the Wikimedia Commons video provider."""
import unittest
from unittest.mock import patch

from src.fetch_stock import (
    _commons_license_allowed,
    _commons_metadata_value,
    _search_wikimedia_videos,
    _download_wikimedia_video,
)


class WikimediaLicenseTests(unittest.TestCase):
    def test_allows_cc0(self):
        self.assertTrue(_commons_license_allowed({
            "LicenseShortName": {"value": "CC0 1.0"},
        }))

    def test_allows_public_domain(self):
        self.assertTrue(_commons_license_allowed({
            "LicenseShortName": {"value": "Public domain"},
        }))

    def test_allows_cc_by(self):
        self.assertTrue(_commons_license_allowed({
            "LicenseShortName": {"value": "CC BY 4.0"},
        }))

    def test_allows_cc_by_from_license_url(self):
        self.assertTrue(_commons_license_allowed({
            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
        }))

    def test_rejects_share_alike(self):
        self.assertFalse(_commons_license_allowed({
            "LicenseShortName": {"value": "CC BY-SA 4.0"},
        }))

    def test_rejects_noncommercial(self):
        self.assertFalse(_commons_license_allowed({
            "LicenseShortName": {"value": "CC BY-NC 4.0"},
        }))

    def test_rejects_no_derivatives(self):
        self.assertFalse(_commons_license_allowed({
            "LicenseShortName": {"value": "CC BY-ND 4.0"},
        }))

    def test_rejects_unknown_license(self):
        self.assertFalse(_commons_license_allowed({
            "LicenseShortName": {"value": "Unknown"},
        }))


    def test_search_returns_only_licensed_video_files_with_creator(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                def metadata(license_name, license_url, artist="Uploader"):
                    return {
                        "LicenseShortName": {"value": license_name},
                        "LicenseUrl": {"value": license_url},
                        "Artist": {"value": artist},
                    }

                return {
                    "query": {
                        "pages": [
                            {
                                "pageid": 101,
                                "title": "File:Archerfish.webm",
                                "imageinfo": [{
                                    "url": "https://upload.wikimedia.org/wikipedia/commons/a/a/Archerfish.webm",
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Archerfish.webm",
                                    "mime": "video/webm",
                                    "mediatype": "VIDEO",
                                    "extmetadata": metadata(
                                        "CC BY 4.0",
                                        "https://creativecommons.org/licenses/by/4.0/",
                                    ),
                                }],
                            },
                            {
                                "pageid": 102,
                                "title": "File:Archerfish-NC.mp4",
                                "imageinfo": [{
                                    "url": "https://upload.wikimedia.org/a/Archerfish-NC.mp4",
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Archerfish-NC.mp4",
                                    "mime": "video/mp4",
                                    "mediatype": "VIDEO",
                                    "extmetadata": metadata(
                                        "CC BY-NC 4.0",
                                        "https://creativecommons.org/licenses/by-nc/4.0/",
                                    ),
                                }],
                            },
                            {
                                "pageid": 103,
                                "title": "File:Archerfish.jpg",
                                "imageinfo": [{
                                    "url": "https://upload.wikimedia.org/a/Archerfish.jpg",
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Archerfish.jpg",
                                    "mime": "image/jpeg",
                                    "mediatype": "BITMAP",
                                    "extmetadata": metadata("CC0 1.0", ""),
                                }],
                            },
                            {
                                "pageid": 104,
                                "title": "File:Archerfish-no-author.mp4",
                                "imageinfo": [{
                                    "url": "https://upload.wikimedia.org/a/Archerfish-no-author.mp4",
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Archerfish-no-author.mp4",
                                    "mime": "video/mp4",
                                    "mediatype": "VIDEO",
                                    "extmetadata": metadata(
                                        "CC BY 4.0",
                                        "https://creativecommons.org/licenses/by/4.0/",
                                        artist="",
                                    ),
                                }],
                            },
                        ]
                    }
                }

        with patch("src.fetch_stock.requests.get", return_value=FakeResponse()) as mocked_get:
            results = _search_wikimedia_videos("Archerfish", limit=10)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["video_id"], "commons:101")
        self.assertEqual(results[0]["author"], "Uploader")
        self.assertEqual(results[0]["license"], "CC BY 4.0")
        self.assertEqual(mocked_get.call_args.kwargs["params"]["gsrnamespace"], 6)
        self.assertIn("Archerfish", mocked_get.call_args.kwargs["params"]["gsrsearch"])

    def test_download_rejects_non_wikimedia_host_without_request(self):
        with patch("src.fetch_stock.requests.get") as mocked_get:
            result = _download_wikimedia_video(
                {"media_url": "https://example.com/video.mp4"},
                __import__("pathlib").Path("should_not_be_created.mp4"),
            )
        self.assertFalse(result)
        mocked_get.assert_not_called()

    def test_metadata_html_is_cleaned(self):
        self.assertEqual(
            _commons_metadata_value({"Artist": {"value": "<a>Jane Doe</a>"}}, "Artist"),
            "Jane Doe",
        )


if __name__ == "__main__":
    unittest.main()
