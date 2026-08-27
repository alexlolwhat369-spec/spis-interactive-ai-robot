"""Try conversation, music selection, and the object game without microphone hardware."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .conversation import OllamaConversationProvider, RuleConversationProvider
    from .music import MusicSelector
    from .robot_runtime import RobotDialogueSession
    from .speech import LocalSpeaker
except ImportError:
    from conversation import OllamaConversationProvider, RuleConversationProvider
    from music import MusicSelector
    from robot_runtime import RobotDialogueSession
    from speech import LocalSpeaker

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the robot logic without microphone hardware.")
    parser.add_argument("--ollama-model", help="Use a model already installed in Ollama.")
    parser.add_argument("--speak", action="store_true", help="Read robot replies aloud using an available local voice.")
    args = parser.parse_args()
    provider = OllamaConversationProvider(args.ollama_model) if args.ollama_model else RuleConversationProvider()
    session = RobotDialogueSession(provider, ROOT / "data" / "object_catalog.json")
    music = MusicSelector.from_file(ROOT / "assets" / "music" / "playlist.json")
    speaker = LocalSpeaker()
    print("Type in English. Type 'quit' to exit.")
    while True:
        message = input("You: ").strip()
        if message.lower() == "quit":
            return
        session_result = session.respond(message)
        result = session_result.conversation
        print(f"Robot [{result.command.reaction}]: {result.command.reply}")
        if args.speak and not speaker.speak(result.command.reply, result.command.reaction):
            print("(No local voice was found. The subtitle remains available.)")
        if result.provider_error:
            print("(Ollama unavailable; local fallback used.)")
        if result.command.action.value == "play_music":
            track = music.choose(requested_category=result.music_category)
            print(f"Music selection: {track.title} ({track.category})")


if __name__ == "__main__":
    main()
