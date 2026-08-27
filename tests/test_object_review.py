from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.object_review import approve_suggestion, load_suggestions, reject_suggestion


class ObjectReviewTests(unittest.TestCase):
    def _write_fixture(self, directory: Path) -> tuple[Path, Path]:
        catalog = directory / "catalog.json"
        queue = directory / "suggestions.jsonl"
        catalog.write_text(
            json.dumps(
                {
                    "questions": {"electronic": {"text": "Is it electronic?", "category": "technology"}},
                    "objects": [
                        {"name": "phone", "category": "technology", "attributes": ["electronic"]},
                        {"name": "book", "category": "school", "attributes": []},
                    ],
                }
            ),
            encoding="utf-8",
        )
        queue.write_text(
            json.dumps(
                {
                    "guessed_object": "phone",
                    "new_object": "desktop computer",
                    "distinguishing_question": "Does it usually use a separate monitor and keyboard?",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return catalog, queue

    def test_approve_adds_a_reviewed_object_and_removes_its_queue_entry(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            catalog, queue = self._write_fixture(Path(temporary_directory))
            approve_suggestion(
                queue,
                catalog,
                1,
                category="technology",
                attributes=["electronic", "desktop_setup"],
                distinguishing_attribute="desktop_setup",
                question_text="Does it usually use a separate monitor and keyboard?",
            )
            stored_catalog = json.loads(catalog.read_text(encoding="utf-8"))

            self.assertEqual(load_suggestions(queue), [])
        self.assertIn("desktop computer", [item["name"] for item in stored_catalog["objects"]])
        self.assertIn("desktop_setup", stored_catalog["questions"])

    def test_reject_removes_only_the_selected_suggestion(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            _, queue = self._write_fixture(Path(temporary_directory))
            rejected = reject_suggestion(queue, 1)

            self.assertEqual(rejected["new_object"], "desktop computer")
            self.assertEqual(load_suggestions(queue), [])
