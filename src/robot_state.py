"""The robot's visible state is controlled by explicit, testable events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Reaction(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HAPPY = "happy"
    PROUD = "proud"
    CONFUSED = "confused"
    HEART = "heart"
    ANNOYED = "annoyed"
    CURIOUS = "curious"


class Action(StrEnum):
    NONE = "none"
    START_GAME = "start_game"
    STOP = "stop"
    PLAY_MUSIC = "play_music"


@dataclass(frozen=True)
class RobotCommand:
    reply: str
    reaction: Reaction
    action: Action = Action.NONE


class RobotController:
    """Maps trusted events to the robot's reaction without profiling visitors."""

    def __init__(self) -> None:
        self.reaction = Reaction.IDLE

    def from_gesture(self, gesture: str) -> RobotCommand:
        commands = {
            "wave": RobotCommand("Hello!", Reaction.HAPPY),
            "thumbs_up": RobotCommand("Awesome!", Reaction.PROUD),
            "peace": RobotCommand("Peace!", Reaction.HAPPY),
            "stop": RobotCommand("Okay, I will wait.", Reaction.LISTENING, Action.STOP),
            "heart": RobotCommand("I love that!", Reaction.HEART),
            "unknown": RobotCommand("I am not sure which gesture that was.", Reaction.CONFUSED),
            "none": RobotCommand("", Reaction.IDLE),
        }
        command = commands.get(gesture, commands["unknown"])
        self.reaction = command.reaction
        return command

    def begin_listening(self) -> RobotCommand:
        self.reaction = Reaction.LISTENING
        return RobotCommand("", self.reaction)

    def begin_thinking(self) -> RobotCommand:
        self.reaction = Reaction.THINKING
        return RobotCommand("", self.reaction)

    def present(self, command: RobotCommand) -> RobotCommand:
        self.reaction = command.reaction
        return command

    def finish_speaking(self) -> RobotCommand:
        self.reaction = Reaction.IDLE
        return RobotCommand("", self.reaction)
