from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.conversation import RuleConversationProvider
from src.robot_runtime import RobotDialogueSession, TurnRoute, answer_from_text


ROOT = Path(__file__).resolve().parents[1]


class InteractionBenchmarkTests(unittest.TestCase):
    def test_forty_common_phrases_reach_the_expected_owner(self) -> None:
        cases = {
            TurnRoute.CONVERSATION: (
                "hello robot",
                "tell me a joke",
                "you are cute",
                "that is interesting",
                "what can you do",
                "tell me about space",
                "I had a difficult day",
                "thank you robot",
            ),
            TurnRoute.MUSIC_REQUEST: (
                "play music",
                "play a song",
                "ok play",
                "please play it",
                "put on something",
                "play energetic music",
                "I want relaxing music",
                "play music for my birthday",
            ),
            TurnRoute.GAME_START: (
                "play Akinator",
                "start Alkinator",
                "play the guessing game",
                "start the object game",
                "twenty questions",
                "can you guess my object",
                "guess what I am thinking",
                "let us play a game",
            ),
            TurnRoute.MUSIC_CONTROL: (
                "pause",
                "pause the music",
                "resume",
                "continue the song",
                "next song",
                "skip track",
                "stop music",
                "turn off the music",
            ),
        }
        with TemporaryDirectory() as directory:
            session = RobotDialogueSession(
                RuleConversationProvider(),
                ROOT / "data" / "object_catalog.json",
                trial_log_path=Path(directory) / "trials.jsonl",
            )
            for expected, phrases in cases.items():
                for phrase in phrases:
                    with self.subTest(phrase=phrase):
                        self.assertEqual(session.route_message(phrase), expected)

    def test_uncertain_game_answer_vocabulary_is_stable(self) -> None:
        cases = {
            "yes": "yes",
            "yeah": "yes",
            "definitely": "yes",
            "probably": "probably",
            "most of the time": "probably",
            "I think so": "probably",
            "maybe": "maybe",
            "I am not sure": "maybe",
            "probably not": "probably_not",
            "not usually": "probably_not",
            "no": "no",
            "not at all": "no",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(answer_from_text(phrase), expected)
