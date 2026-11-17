import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.generate_script import (
    _get_recent_animals,
    _normalize_animal,
    _save_last_animal,
)


class SpeciesRotationTests(unittest.TestCase):
    def test_scientific_name_is_the_canonical_species_identity(self):
        first = _normalize_animal("Archerfish (Toxotes jaculatrix)")
        synonym = _normalize_animal("Banded archerfish (Toxotes jaculatrix)")
        self.assertEqual(first, synonym)

    def test_archerfish_common_name_maps_to_its_scientific_identity(self):
        self.assertEqual(
            _normalize_animal("Archerfish"),
            _normalize_animal("Archerfish (Toxotes jaculatrix)"),
        )

    def test_species_history_is_not_truncated_to_twenty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "cosmic_curious_last_animal.json"
            with patch(
                "src.generate_script._last_animal_path",
                return_value=history_path,
            ):
                for index in range(30):
                    _save_last_animal("test_channel", f"Animal {index}")

                history = _get_recent_animals("test_channel")

            self.assertEqual(len(history), 30)
            self.assertEqual(history[0], "Animal 0")
            self.assertEqual(history[-1], "Animal 29")


if __name__ == "__main__":
    unittest.main()
