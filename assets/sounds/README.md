# Local sound effects

`interactive_robot.py` plays `mohan_whistle.mp3` once when the trained `mohan`
gesture activates. MP3 effects are intentionally excluded from Git so only audio
that the project has permission to redistribute is published.

To use a different authorized file:

```powershell
python src/interactive_robot.py --mohan-sound "C:\path\to\effect.mp3"
```
