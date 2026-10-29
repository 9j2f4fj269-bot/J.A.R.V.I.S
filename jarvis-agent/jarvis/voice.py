"""Optional voice output. Uses whatever TTS binary the OS already has.

Speech-to-text is intentionally left to the platform (e.g. a Whisper API call) so
this package stays dependency-free; `listen()` explains the hook if you add one.
"""

from __future__ import annotations

import re
import shutil
import subprocess

CANDIDATES = [
    ("say", ["say", "-v", "Daniel"]),          # macOS
    ("espeak-ng", ["espeak-ng", "-v", "en-gb"]),  # Linux
    ("espeak", ["espeak", "-v", "en-gb"]),
    ("spd-say", ["spd-say", "-w"]),
    ("festival", ["festival", "--tts"]),
]


def find_engine() -> list[str] | None:
    for binary, command in CANDIDATES:
        if shutil.which(binary):
            return command
    return None


def available() -> bool:
    return find_engine() is not None


def _plain(text: str) -> str:
    text = re.sub(r"```.*?```", " code block ", text, flags=re.S)
    text = re.sub(r"[*_#`>]+", "", text)
    text = re.sub(r"https?://\S+", "link", text)
    return re.sub(r"\s+", " ", text).strip()


def speak(text: str, max_chars: int = 1200) -> bool:
    """Speak text aloud. Returns False when no engine is installed."""
    command = find_engine()
    if not command:
        return False
    spoken = _plain(text)[:max_chars]
    if not spoken:
        return False
    try:
        if command[0] == "festival":
            subprocess.run(command + [spoken], check=False, timeout=120)
        elif command[0] == "spd-say":
            subprocess.run(command + [spoken], check=False, timeout=120)
        else:
            subprocess.run(command + [spoken], check=False, timeout=120)
        return True
    except Exception:
        return False


def listen() -> None:
    """Hook for speech input.

    Record audio with `arecord`/`sox`, then POST it to a speech-to-text endpoint
    (OpenAI /audio/transcriptions or a local whisper.cpp server) and pass the text
    into Agent.run(). Kept as a stub so no microphone dependency is forced.
    """
    raise NotImplementedError(
        "Wire a recorder + STT endpoint here, then feed the transcript to Agent.run()."
    )
