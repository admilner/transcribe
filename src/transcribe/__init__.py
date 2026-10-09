"""Transcribe speech from the Windows default communications microphone with OpenAI Whisper."""

import argparse
import queue
import sys
import threading

import numpy as np
import sounddevice as sd

WHISPER_RATE = 16_000
BLOCK_SECONDS = 0.1


def default_communications_mic() -> int | None:
    """Return the sounddevice index of the Windows default communications capture device."""
    from pycaw.constants import EDataFlow, ERole
    from pycaw.utils import AudioUtilities

    try:
        endpoint = AudioUtilities.GetDeviceEnumerator().GetDefaultAudioEndpoint(
            EDataFlow.eCapture.value, ERole.eCommunications.value
        )
        name = AudioUtilities.CreateDevice(endpoint).FriendlyName
    except Exception:
        return None

    wasapi = next(h for h in sd.query_hostapis() if h["name"] == "Windows WASAPI")
    for index in wasapi["devices"]:
        if sd.query_devices(index)["name"] == name:
            return index
    return None


def record_utterances(device, threshold: float, silence: float, max_seconds: float):
    """Yield 16 kHz mono float32 arrays, one per utterance, split on pauses in speech."""
    info = sd.query_devices(device, "input")
    rate = int(info["default_samplerate"])
    blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time, status):
        blocks.put(indata.mean(axis=1).copy())

    silent_blocks_to_end = int(silence / BLOCK_SECONDS)
    max_blocks = int(max_seconds / BLOCK_SECONDS)
    with sd.InputStream(
        device=device,
        channels=info["max_input_channels"],
        samplerate=rate,
        blocksize=int(rate * BLOCK_SECONDS),
        dtype="float32",
        callback=callback,
    ):
        print(f"Listening on {info['name']} (Ctrl+C to stop)...", file=sys.stderr)
        speech: list[np.ndarray] = []
        silent = 0
        previous = None
        while True:
            block = blocks.get()
            loud = np.sqrt(np.mean(block**2)) >= threshold
            if not speech:
                if loud:
                    # Keep the block before speech started so the first syllable isn't clipped.
                    speech = [previous, block] if previous is not None else [block]
                previous = block
                continue
            speech.append(block)
            silent = 0 if loud else silent + 1
            if silent >= silent_blocks_to_end or len(speech) >= max_blocks:
                yield resample(np.concatenate(speech), rate)
                speech, silent, previous = [], 0, None


def resample(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == WHISPER_RATE:
        return audio
    target = np.linspace(0, len(audio), int(len(audio) * WHISPER_RATE / rate), endpoint=False)
    return np.interp(target, np.arange(len(audio)), audio).astype(np.float32)


def make_writer(args):
    """Return a function that delivers one piece of transcribed text to the chosen output."""
    if args.to_keyboard:
        from pynput.keyboard import Controller

        keyboard = Controller()
        return lambda text: keyboard.type(text + " ")
    if args.to_file:
        out = open(args.to_file, "a", encoding="utf-8")

        def write_file(text):
            out.write(text + "\n")
            out.flush()

        return write_file
    return lambda text: print(text, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--to-keyboard",
        action="store_true",
        help="type the text as keystrokes into the focused window",
    )
    output.add_argument("--to-file", metavar="PATH", help="append the text to a file")
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model: tiny, base, small, medium, large, turbo, or *.en variants (default: base)",
    )
    parser.add_argument("--language", help="spoken language code, e.g. en (default: auto-detect)")
    parser.add_argument(
        "--device",
        type=int,
        help="sounddevice input index (default: Windows default communications mic)",
    )
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="RMS level that counts as speech (default: 0.01)",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=0.8,
        help="seconds of silence that end an utterance (default: 0.8)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=30,
        help="longest utterance before forcing a transcription (default: 30)",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    import torch
    import whisper

    device = args.device if args.device is not None else default_communications_mic()
    gpu = torch.cuda.is_available()
    print(f"Loading Whisper model '{args.model}'...", file=sys.stderr)
    model = whisper.load_model(args.model, device="cuda" if gpu else "cpu")
    write = make_writer(args)

    # Transcribe on a worker thread so recording never stalls while Whisper is busy.
    utterances: queue.Queue[np.ndarray] = queue.Queue()

    def transcribe_forever():
        while True:
            result = model.transcribe(utterances.get(), language=args.language, fp16=gpu)
            text = result["text"].strip()
            if text:
                write(text)

    threading.Thread(target=transcribe_forever, daemon=True).start()
    try:
        for audio in record_utterances(device, args.threshold, args.silence, args.max_seconds):
            utterances.put(audio)
    except KeyboardInterrupt:
        pass
