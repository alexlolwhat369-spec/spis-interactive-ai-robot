from __future__ import annotations

import unittest

from src.conversation import ConversationResult, OllamaConversationProvider, RuleConversationProvider, explicit_action_result
from src.robot_state import Action, Reaction, RobotCommand


class ConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = RuleConversationProvider()

    def test_game_request_starts_game(self) -> None:
        result = self.provider.respond("Can we play a guessing game?")
        self.assertEqual(result.command.action, Action.START_GAME)
        self.assertEqual(result.command.reaction, Reaction.PROUD)

    def test_music_request_uses_explicit_category(self) -> None:
        result = self.provider.respond("Play energetic music")
        self.assertEqual(result.command.action, Action.PLAY_MUSIC)
        self.assertEqual(result.music_category, "energetic")

    def test_explicit_greeting_keeps_a_happy_reaction(self) -> None:
        original = ConversationResult(command=RobotCommand("Hello!", Reaction.CONFUSED))
        result = OllamaConversationProvider._apply_explicit_reaction_rules("Hello robot", original)
        self.assertEqual(result.command.reaction, Reaction.HAPPY)

    def test_game_command_is_deterministic_before_ollama(self) -> None:
        for request in ("Can we play a guessing game?", "Let us play a game", "Can we play twenty questions?"):
            with self.subTest(request=request):
                result = explicit_action_result(request)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.command.action, Action.START_GAME)
                self.assertIn("I will ask questions and try to guess it", result.command.reply)

    def test_explicit_conversation_phrases_choose_visible_reactions_offline(self) -> None:
        cases = {
            "You are so cute": Reaction.HAPPY,
            "You are stupid": Reaction.ANNOYED,
            "I do not understand": Reaction.CONFUSED,
            "That is interesting": Reaction.CURIOUS,
        }

        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(self.provider.respond(phrase).command.reaction, expected)

    def test_explicit_interest_overrides_an_ordinary_ollama_reaction(self) -> None:
        original = ConversationResult(command=RobotCommand("A reply.", Reaction.SPEAKING))
        result = OllamaConversationProvider._apply_explicit_reaction_rules("Wow, interesting!", original)

        self.assertEqual(result.command.reaction, Reaction.CURIOUS)
