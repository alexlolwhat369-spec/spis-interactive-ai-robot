"""Tests for the reaction sound research catalog."""

from __future__ import annotations

import unittest

from src.build_reaction_sound_variants import REFERENCE_SETS, VARIANT_TITLES, validate_catalog


class ReactionSoundLabTests(unittest.TestCase):
    def test_catalog_covers_every_reaction_except_mohan(self) -> None:
        validate_catalog()
        self.assertEqual(len(REFERENCE_SETS), 11)
        self.assertNotIn("mohan", REFERENCE_SETS)

    def test_each_reaction_has_ten_unique_references_and_three_variants(self) -> None:
        for reaction, references in REFERENCE_SETS.items():
            with self.subTest(reaction=reaction):
                self.assertEqual(len(references), 10)
                self.assertEqual(len(set(references)), 10)
                self.assertEqual(len(VARIANT_TITLES[reaction]), 3)


if __name__ == "__main__":
    unittest.main()
