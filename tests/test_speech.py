from __future__ import annotations

import unittest

from src.speech import (
    EMPHATIC_STYLE,
    NEUTRAL_STYLE,
    QUESTION_STYLE,
    clean_spoken_text,
    plan_speech,
    preferred_windows_voice,
)


class SpeechPlanTests(unittest.TestCase):
    def test_proud_delivery_is_more_energetic_than_neutral(self) -> None:
        plan = plan_speech("You solved it.", "proud")

        self.assertEqual(len(plan), 1)
        self.assertLess(plan[0].style.length_scale, NEUTRAL_STYLE.length_scale)

    def test_exclamation_is_emphasized_without_changing_other_sentences(self) -> None:
        plan = plan_speech("I see it. Great work!", "heart")

        self.assertEqual([segment.text for segment in plan], ["I see it.", "Great work!"])
        self.assertEqual(plan[-1].style, EMPHATIC_STYLE)
        self.assertNotEqual(plan[0].style, EMPHATIC_STYLE)

    def test_question_uses_clearer_question_delivery(self) -> None:
        plan = plan_speech("Are you ready?", "happy")

        self.assertEqual(plan[0].style, QUESTION_STYLE)

    def test_empty_reply_has_no_audio_segments(self) -> None:
        self.assertEqual(plan_speech("   ", "happy"), [])

    def test_display_markup_is_not_read_aloud(self) -> None:
        text = "**Great!** Read [the guide](https://example.com), not `code`."

        self.assertEqual(clean_spoken_text(text), "Great! Read the guide, not code.")

    def test_windows_voice_prefers_a_natural_installed_voice(self) -> None:
        installed = ["Microsoft David Desktop", "Microsoft Aria Natural", "Microsoft Zira Desktop"]

        self.assertEqual(preferred_windows_voice(installed), "Microsoft Aria Natural")
        self.assertEqual(preferred_windows_voice(installed, "Microsoft Zira Desktop"), "Microsoft Zira Desktop")
