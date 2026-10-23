"""
Non-blocking Background Voice Command Listener & Intent Parser for i-am-ditto.

Listens continuously for natural phrases such as:
- "Hey I want to become Spider-Man"
- "Turn me into Iron Man"
- "Change my face to Elon Musk"
- "Transform into Joker"
- "Clear filter" / "Reset face"
"""

import re
import threading
import time
from typing import Callable, Optional, Tuple


TRIGGER_PATTERNS = [
    r"(?:i\s+want\s+to\s+become|wanna\s+become|want\s+to\s+be|become\s+a|become\s+an|become)\s+(.+)",
    r"(?:turn\s+me\s+into|transform\s+me\s+into|transform\s+into)\s+(.+)",
    r"(?:change\s+my\s+face\s+to|change\s+to|switch\s+to|make\s+me)\s+(.+)",
    r"(?:ditto\s+become|hey\s+ditto\s+become|ditto)\s+(.+)",
]

CLEAR_WORDS = {"clear", "reset", "normal", "none", "off", "remove filter", "clear filter", "stop filter"}


def parse_voice_intent(transcript: str) -> Tuple[Optional[str], str]:
    """
    Parse spoken text into (action, character_name).
    Returns:
      - ("become", "<character_name>")
      - ("clear", "")
      - (None, "") if no trigger matched
    """
    cleaned = transcript.strip().lower()
    cleaned = re.sub(r"[^\w\s\-]", "", cleaned).strip()
    if not cleaned:
        return None, ""

    if cleaned in CLEAR_WORDS or any(w in cleaned for w in ("clear filter", "remove filter", "normal face", "reset face")):
        return "clear", ""

    for pattern in TRIGGER_PATTERNS:
        match = re.search(pattern, cleaned)
        if match:
            target = match.group(1).strip()
            # Strip trailing filler words
            target = re.sub(r"\b(please|now|right now|filter|mask|face)\b$", "", target).strip()
            target = re.sub(r"^(a|an|the)\s+", "", target).strip()
            if target in CLEAR_WORDS:
                return "clear", ""
            if len(target) >= 2:
                return "become", target

    return None, ""


class VoiceCommandListener:
    """Continuously listens to the microphone in a background daemon thread."""

    def __init__(
        self,
        on_command: Callable[[str, str], None],
        on_transcript: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.on_command = on_command
        self.on_transcript = on_transcript
        self.on_status = on_status
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _listen_loop(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError:
            if self.on_status:
                self.on_status("Voice module not installed (Press [T] to type)")
            return

        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.65

        try:
            mic = sr.Microphone()
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
            if self.on_status:
                self.on_status("Mic Ready: Say 'become <character>'")
        except Exception:
            if self.on_status:
                self.on_status("No Mic Detected — Press [T] to type character")
            return

        while self._running:
            try:
                with sr.Microphone() as source:
                    audio = recognizer.listen(source, timeout=3.0, phrase_time_limit=5.0)
                text = recognizer.recognize_google(audio)
                if text and self.on_transcript:
                    self.on_transcript(text)

                action, target = parse_voice_intent(text)
                if action:
                    self.on_command(action, target)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception:
                time.sleep(0.5)
