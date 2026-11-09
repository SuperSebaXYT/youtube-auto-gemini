"""Offline tests for the Wikimedia Commons video provider."""
import unittest

from src.fetch_stock import _commons_license_allowed, _commons_metadata_value


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

    def test_metadata_html_is_cleaned(self):
        self.assertEqual(
            _commons_metadata_value({"Artist": {"value": "<a>Jane Doe</a>"}}, "Artist"),
            "Jane Doe",
        )


if __name__ == "__main__":
    unittest.main()
