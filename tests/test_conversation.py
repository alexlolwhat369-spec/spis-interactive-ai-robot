from __future__ import annotations

import unittest

from src.conversation import (
    ConversationResult,
    OllamaConversationProvider,
    RuleConversationProvider,
    explicit_action_result,
    is_game_request,
    is_music_composition_request,
    is_music_request,
    music_control_action,
    _valid_game_question,
)
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

    def test_music_controls_are_deterministic_and_do_not_start_the_game(self) -> None:
        cases = {
            "stop the music": Action.STOP_MUSIC,
            "and to stop the music": Action.STOP_MUSIC,
            "I said stop the music": Action.STOP_MUSIC,
            "could you pause the music please": Action.PAUSE_MUSIC,
            "pause": Action.PAUSE_MUSIC,
            "resume music": Action.RESUME_MUSIC,
            "next song": Action.NEXT_MUSIC,
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(music_control_action(message), expected)
                result = explicit_action_result(message)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.command.action, expected)
                self.assertFalse(is_game_request(message))

    def test_discussing_a_music_control_does_not_execute_it(self) -> None:
        self.assertIsNone(music_control_action("How do I stop the music player?"))
        self.assertIsNone(music_control_action("Tell me why people pause music."))

    def test_music_mood_phrases_choose_the_expected_category(self) -> None:
        cases = {
            "Play something because I feel stressed": "calm",
            "Play romantic music": "warm",
            "Play a fun happy song": "happy",
            "Play workout music": "energetic",
            "Play music for my birthday": "celebration",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                result = self.provider.respond(message)
                self.assertEqual(result.command.action, Action.PLAY_MUSIC)
                self.assertEqual(result.music_category, expected)

    def test_capability_question_never_reaches_the_small_model_or_invents_tools(self) -> None:
        result = explicit_action_result("What can you do?")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("chat with you", result.command.reply)
        self.assertIn("installed music", result.command.reply)
        self.assertNotIn("create music", result.command.reply)

    def test_composing_music_is_not_confused_with_playing_an_installed_song(self) -> None:
        self.assertTrue(is_music_composition_request("Can you create a new song for me?"))
        self.assertFalse(is_music_request("Can you create a new song for me?"))
        self.assertFalse(is_music_request("I like this song"))

        result = explicit_action_result("Can you create a new song for me?")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("cannot create", result.command.reply)
        self.assertEqual(result.command.action, Action.NONE)

    def test_requested_follow_up_question_is_added_when_small_model_omits_it(self) -> None:
        original = ConversationResult(RobotCommand("Mars has Olympus Mons.", Reaction.CONFUSED))

        repaired = OllamaConversationProvider._guard_reply_content(
            "Tell me a Mars fact and ask me a question.", original
        )

        self.assertIn("?", repaired.command.reply)
        self.assertEqual(repaired.command.reaction, Reaction.SPEAKING)

    def test_compound_compliment_and_request_acknowledges_both_parts(self) -> None:
        original = ConversationResult(RobotCommand("Why did the robot cross the road?", Reaction.HAPPY))

        repaired = OllamaConversationProvider._guard_reply_content(
            "You are cute, and tell me a short joke.", original
        )

        self.assertTrue(repaired.command.reply.startswith("Thank you!"))
        self.assertIn("robot", repaired.command.reply)

    def test_unsolicited_game_invitation_is_removed_from_normal_conversation(self) -> None:
        original = ConversationResult(
            RobotCommand("That is interesting! Would you like to play the object guessing game?", Reaction.CURIOUS)
        )

        repaired = OllamaConversationProvider._guard_reply_content("Wow, interesting", original)

        self.assertEqual(repaired.command.reply, "That is interesting!")

    def test_short_play_command_is_music_and_never_a_game(self) -> None:
        for request in ("ok play", "okay play it", "play", "please play", "play a song"):
            with self.subTest(request=request):
                self.assertTrue(is_music_request(request))
                self.assertFalse(is_game_request(request))
                result = explicit_action_result(request)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.command.action, Action.NONE)
                self.assertEqual(result.command.reaction, Reaction.LISTENING)
                self.assertIn("What kind of music", result.command.reply)

    def test_explicit_greeting_keeps_a_happy_reaction(self) -> None:
        original = ConversationResult(command=RobotCommand("Hello!", Reaction.CONFUSED))
        result = OllamaConversationProvider._apply_explicit_reaction_rules("Hello robot", original)
        self.assertEqual(result.command.reaction, Reaction.HAPPY)

    def test_game_command_is_deterministic_before_ollama(self) -> None:
        for request in (
            "Can we play a guessing game?",
            "Let us play a game",
            "Can we play twenty questions?",
            "Can we play Alkinator?",
            "Start a kinator",
            "Can you guess what I'm thinking?",
            "Let us try the object game",
        ):
            with self.subTest(request=request):
                result = explicit_action_result(request)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.command.action, Action.START_GAME)
                self.assertIn("I will ask questions and try to guess it", result.command.reply)

    def test_game_request_does_not_trigger_on_an_incidental_game_reference(self) -> None:
        self.assertFalse(is_game_request("That game was interesting yesterday."))

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

    def test_system_prompt_is_never_dropped_from_long_conversation_history(self) -> None:
        provider = OllamaConversationProvider("test-model")
        for index in range(8):
            provider.history.extend(
                [
                    {"role": "user", "content": f"question {index}"},
                    {"role": "assistant", "content": f"answer {index}"},
                ]
            )

        messages = provider._messages_for("What did we discuss?")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Address every distinct question", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "What did we discuss?"})
        self.assertLessEqual(len(messages), 10)

    def test_small_model_cannot_invent_a_game_action(self) -> None:
        invented = ConversationResult(RobotCommand("Let us play.", Reaction.HAPPY, Action.START_GAME))

        guarded = OllamaConversationProvider._guard_model_action("Tell me about Mars", invented)

        self.assertEqual(guarded.command.action, Action.NONE)

    def test_small_model_cannot_invent_music_controls(self) -> None:
        invented = ConversationResult(RobotCommand("Paused.", Reaction.OK, Action.PAUSE_MUSIC))

        guarded = OllamaConversationProvider._guard_model_action("Tell me about Mars", invented)
        allowed = OllamaConversationProvider._guard_model_action("pause music", invented)

        self.assertEqual(guarded.command.action, Action.NONE)
        self.assertEqual(allowed.command.action, Action.PAUSE_MUSIC)

    def test_semantic_game_answer_requires_high_confidence(self) -> None:
        provider = OllamaConversationProvider("test-model")
        provider._structured_request = lambda *args, **kwargs: (  # type: ignore[method-assign]
            {"answer": "yes", "confidence": 0.91},
            "{}",
        )

        answer = provider.interpret_game_answer(
            "I use it to move the cursor around",
            "Is it used to move a computer pointer?",
            [],
        )

        self.assertEqual(answer, "yes")

        provider._structured_request = lambda *args, **kwargs: (  # type: ignore[method-assign]
            {"answer": "yes", "confidence": 0.45},
            "{}",
        )
        self.assertIsNone(provider.interpret_game_answer("Maybe something else", "Is it electronic?", []))

        provider._structured_request = lambda *args, **kwargs: (  # type: ignore[method-assign]
            {"answer": "no", "confidence": 0.99},
            "{}",
        )
        self.assertIsNone(provider.interpret_game_answer("I like pizza", "Does it have a screen?", []))

    def test_generated_game_question_validator_rejects_role_changes_and_double_questions(self) -> None:
        self.assertTrue(_valid_game_question("Does it normally have a screen?"))
        self.assertFalse(_valid_game_question("Guess my object?"))
        self.assertFalse(_valid_game_question("Is it electronic? Does it have a screen?"))
        self.assertFalse(_valid_game_question("Is it a phone or a tablet?"))
