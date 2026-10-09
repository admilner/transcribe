# transcribe

This is a simple script to transcribe from my microphone on Windows to text.

Audio from the Windows default *communications* microphone is split into utterances on pauses
and each utterance is transcribed with [OpenAI Whisper](https://github.com/openai/whisper).

## Usage

```
uv run transcribe                     # print text to stdout
uv run transcribe --to-keyboard       # type text into the focused window
uv run transcribe --to-file out.txt   # append text to out.txt
```

Stop with Ctrl+C. Other options: `--model` (default `base`; try `small` or `base.en` for
better accuracy), `--language en`, `--device N` (see `--list-devices`), and `--threshold`
/ `--silence` to tune how speech is detected. The first run downloads the model.
