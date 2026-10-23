# `i-am-ditto` — AI Voice-Activated Face & Body AR Transformer

> **Become anyone or anything in real time.**  
> `i-am-ditto` tracks your human **Face (478 3D landmarks)**, **Body Pose (33 landmarks)**, and **Hands (42 landmarks)** with futuristic neon-green MediaPipe sticks, listens to your voice (*"Hey, I want to become Spider-Man"*), resolves or generates the character online via a 3-Tier AI Agent, and warps the character's 3D triangle mesh onto your live face like an Instagram AR filter.

---

## Key Features

1. **Neon-Green Cybernetic Body & Face Scanner (`MediaPipe Holistic`)**
   - Simultaneous 3D tracking of body skeleton sticks, finger joints, and 478 facial landmarks.
2. **Voice-Activated Transformation (`SpeechRecognition`)**
   - Speak naturally into your microphone:
     - *"Hey I want to become Spider-Man"*
     - *"Turn me into Iron Man"*
     - *"Become Elon Musk"*
     - *"Transform into Cyberpunk Samurai"*
     - *"Clear filter"*
3. **3-Tier AI Avatar Resolver (`ai/avatar_resolver.py`)**
   - **Tier 1 (Local Cache):** Instant `<10ms` loading from `avatars/<slug>/` with built-in starter masks (*Spider-Man, Iron Man, Batman, Joker*).
   - **Tier 2 (Online Search):** Searches the web for clean front-facing faces/masks and validates landmark quality with MediaPipe.
   - **Tier 3 (Generative AI Fallback):** Synthesizes a symmetrical front-facing portrait mask on the fly if no clean match is found online.
4. **Dual 3D Mesh Warping Engine (`core/mesh_warper.py`)**
   - **Human Faces:** Extracts real facial landmarks from photos so eyes, nose, lips, and jaw align and move with your expressions.
   - **Superhero Masks / Helmets:** Maps masks onto a **Canonical 468-Aligned UV Topology** so full-face helmets (like Spider-Man or Iron Man) warp smoothly in 3D as you rotate your head or open your mouth.

---

## Quick Start

```bash
# 1. Activate the Python 3.11 virtual environment
.\.venv\Scripts\activate

# 2. Launch i-am-ditto (Real-time Webcam + Voice Listener)
python main.py

# Or start directly transformed into a character:
python main.py --become "spiderman"
```

---

## Live Controls (Inside Camera Window)

| Voice / Key | Action |
| :--- | :--- |
| 🎙️ **"Hey I want to become `<name>`"** | AI resolves `<name>` and warps it onto your face |
| **`[T]`** | Type any character name on screen and press `Enter` |
| **`[1]` / `[2]` / `[3]` / `[4]`** | Instant presets: `Spider-Man`, `Iron Man`, `Batman`, `Joker` |
| **`[S]`** | Toggle neon-green Body & Hand skeleton sticks ON/OFF |
| **`[F]`** | Toggle neon-green Face Mesh wireframe ON/OFF |
| **`[E]`** | Toggle real Eye & Mouth cutouts ON/OFF |
| **`[C]`** | Clear filter and return to Scanner Mode |
| **`[Q]` / `ESC`** | Quit |

---

## Project Architecture

```text
i-am-ditto/
├── main.py                  # Main real-time webcam app & CLI entry point
├── core/
│   ├── holistic_tracker.py  # MediaPipe Holistic scanner + Neon-Green sticks + HUD
│   └── mesh_warper.py       # 84-node Delaunay 3D triangle mesh warper + Canonical UV
├── ai/
│   ├── avatar_resolver.py   # 3-Tier AI Resolver (Cache -> Online Search -> AI Gen)
│   └── voice_listener.py    # Background microphone listener & intent parser
└── avatars/                 # Auto-populated local cache of textures & landmark maps
```
