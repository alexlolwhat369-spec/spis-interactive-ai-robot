# Local sound effects

The robot plays a mapped sound once when a trained gesture becomes active:

- `thumbs_up_reaction.mp3` is mapped to `thumbs_up`.
- `heart_reaction.wav` is mapped to `heart`.
- `peace_reaction.wav` is mapped to `peace`.
- `ok_reaction.mp3` is mapped to `ok`.
- `angry_reaction.mp3` is mapped to the angry `middle_finger` reaction.
- `mohan_whistle.mp3` is mapped to `mohan`.

The heart clip is made from the first 7 seconds of a user-provided recording.
The peace clip uses the user-provided party-horn-soundeffectsfactory.mp3,
capped at 8 seconds (the full recording is shorter). It replaces Tapion Ocarina.
The selected reaction files belong alongside the code so a checkout does not
silently lose its effects. Third-party recordings retain their original rights;
their inclusion does not grant a separate redistribution license.

Approved sources: heart = Marvin Gaye - Let's Get It On (first 7 seconds);
peace = party-horn-soundeffectsfactory (3.811 seconds); OK = Ding Sound Effect 4;
thumbs-up = zec53-business-upbeat-short-logo-10-sec-271916;
angry = Roblox angry sound effect; Mohan = Josh Hutcherson Whistle.
The old audition samples and reaction sound lab have been removed. Only the
six selected clips above are shipped; stop has no assigned sound.

Only one effect plays at a time. Its visual reaction uses the decoded file
duration, so the face and sound start and finish together. Holding a gesture
does not loop its sound; a different gesture replaces the active effect.

To use a different authorized file:

```powershell
python src/interactive_robot.py --thumbs-up-sound "C:\path\to\thumbs-up.mp3" --heart-sound "C:\path\to\heart.wav" --peace-sound "C:\path\to\peace.wav" --ok-sound "C:\path\to\ok.mp3" --angry-sound "C:\path\to\angry.mp3" --mohan-sound "C:\path\to\mohan.mp3"
```
