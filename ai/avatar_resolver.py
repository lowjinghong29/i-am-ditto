"""
3-Tier AI Avatar Resolver for i-am-ditto.

When a user says "Hey I want to become <character>", this module resolves the character into a
3D-warpable AvatarAsset (BGRA texture + 84 source landmarks) without blocking the camera loop:

- Tier 1: Local Disk Cache (`avatars/<slug>/texture.png` + `landmarks.npy`) -> <10ms
- Tier 2: Online Image Search (DuckDuckGo Image Search + MediaPipe Face Quality Ranking) -> 1-2s
- Tier 3: Generative AI Fallback (Pollinations Flux front-facing mask generator) -> 2-4s
- Tier 4: Procedural Vector Mask Fallback (guarantees offline resilience)
"""

import os
import re
import threading
import urllib.parse
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import cv2
import numpy as np
import requests

from core.mesh_warper import (
    CANONICAL_UV_ARRAY,
    AvatarAsset,
    extract_avatar_landmarks,
)


AVATARS_DIR = Path(__file__).resolve().parent.parent / "avatars"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "avatar"


class AvatarResolver:
    """Thread-safe asynchronous 3-tier AI resolver for any requested character."""

    def __init__(self, on_status_update: Optional[Callable[[str], None]] = None):
        self.on_status_update = on_status_update
        self._memory_cache: Dict[str, AvatarAsset] = {}
        self._lock = threading.Lock()
        AVATARS_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_builtin_starters()

    def _set_status(self, message: str) -> None:
        if self.on_status_update:
            self.on_status_update(message)

    def resolve_async(
        self,
        character_query: str,
        on_ready: Callable[[AvatarAsset], None],
    ) -> None:
        """Resolve a character in a background daemon thread so webcam FPS never drops."""
        worker = threading.Thread(
            target=self._resolve_worker,
            args=(character_query, on_ready),
            daemon=True,
        )
        worker.start()

    def _resolve_worker(
        self,
        character_query: str,
        on_ready: Callable[[AvatarAsset], None],
    ) -> None:
        clean_name = character_query.strip()
        slug = slugify(clean_name)

        # --- TIER 1: Memory & Local Disk Cache ---
        with self._lock:
            if slug in self._memory_cache:
                self._set_status(f"Transformed: {clean_name.title()} (Cached)")
                on_ready(self._memory_cache[slug])
                return

        cached_asset = self._load_from_disk(clean_name, slug)
        if cached_asset is not None:
            with self._lock:
                self._memory_cache[slug] = cached_asset
            self._set_status(f"Transformed: {clean_name.title()} (Local Library)")
            on_ready(cached_asset)
            return

        # --- TIER 2: Online Search (DuckDuckGo Images + Face/Mask Scoring) ---
        self._set_status(f"AI Agent: Searching online for '{clean_name}'...")
        online_img = self._search_online_face(clean_name)

        # --- TIER 3: Generative AI Front-Facing Mask Fallback ---
        if online_img is None:
            self._set_status(f"AI Agent: Generating 3D mask for '{clean_name}'...")
            online_img = self._generate_ai_face(clean_name)

        # --- TIER 4: Procedural Hero Mask Fallback (if completely offline) ---
        if online_img is None:
            self._set_status(f"AI Agent: Synthesizing mask for '{clean_name}'...")
            online_img = self._create_procedural_mask(clean_name)

        # Extract 468/84 landmarks (Human FaceMesh or Canonical Mask UV)
        self._set_status(f"AI Agent: Mapping 3D mesh for '{clean_name}'...")
        pts, is_canonical = extract_avatar_landmarks(online_img)

        asset = AvatarAsset(
            name=clean_name.title(),
            slug=slug,
            texture_bgra=online_img,
            source_points=pts,
            is_canonical_mask=is_canonical,
        )

        self._save_to_disk(asset)
        with self._lock:
            self._memory_cache[slug] = asset

        mode_tag = "Mask UV" if is_canonical else "FaceMesh"
        self._set_status(f"Active: {asset.name} ({mode_tag})")
        on_ready(asset)

    def _load_from_disk(self, name: str, slug: str) -> Optional[AvatarAsset]:
        folder = AVATARS_DIR / slug
        tex_path = folder / "texture.png"
        lm_path = folder / "landmarks.npy"
        meta_path = folder / "meta.txt"

        if tex_path.exists() and lm_path.exists():
            img = cv2.imread(str(tex_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                pts = np.load(str(lm_path)).astype(np.float32)
                is_canonical = False
                if meta_path.exists():
                    is_canonical = "canonical=True" in meta_path.read_text(encoding="utf-8")
                return AvatarAsset(
                    name=name.title(),
                    slug=slug,
                    texture_bgra=img,
                    source_points=pts,
                    is_canonical_mask=is_canonical,
                )
        return None

    def _save_to_disk(self, asset: AvatarAsset) -> None:
        folder = AVATARS_DIR / asset.slug
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "texture.png"), asset.texture_bgra)
        np.save(str(folder / "landmarks.npy"), asset.source_points)
        (folder / "meta.txt").write_text(
            f"name={asset.name}\ncanonical={asset.is_canonical_mask}\n",
            encoding="utf-8",
        )

    def _search_online_face(self, character_name: str) -> Optional[np.ndarray]:
        """Search online for a clean front-facing character face or mask."""
        try:
            from duckduckgo_search import DDGS

            queries = [
                f"{character_name} front facing face mask cosplay portrait",
                f"{character_name} face closeup front portrait",
            ]
            candidates = []
            with DDGS() as ddgs:
                for q in queries:
                    results = list(ddgs.images(q, max_results=6))
                    for r in results:
                        url = r.get("image")
                        if url:
                            candidates.append(url)
                    if candidates:
                        break

            best_fallback = None
            for url in candidates[:6]:
                img = self._download_image_bgra(url)
                if img is None:
                    continue
                h, w = img.shape[:2]
                if h < 180 or w < 180:
                    continue

                # Resize to standard 512px max dimension for fast warping
                img = self._normalize_size(img, 512)
                pts, is_canonical = extract_avatar_landmarks(img)
                if not is_canonical:
                    # Found a real human/humanoid face with verified MediaPipe landmarks!
                    return img
                if best_fallback is None:
                    best_fallback = img

            return best_fallback
        except Exception:
            return None

    def _generate_ai_face(self, character_name: str) -> Optional[np.ndarray]:
        """Generate a symmetrical front-facing character portrait/mask via AI endpoint."""
        prompt = (
            f"front facing symmetrical portrait face mask of {character_name}, "
            "looking straight at camera, centered full face, studio lighting, high detail"
        )
        encoded = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&nologo=true&seed=42"
        img = self._download_image_bgra(url, timeout=8.0)
        if img is not None:
            return self._normalize_size(img, 512)
        return None

    def _download_image_bgra(self, url: str, timeout: float = 4.5) -> Optional[np.ndarray]:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code != 200 or not resp.content:
                return None
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            return img
        except Exception:
            return None

    @staticmethod
    def _normalize_size(img_bgra: np.ndarray, target_size: int = 512) -> np.ndarray:
        h, w = img_bgra.shape[:2]
        scale = target_size / float(max(h, w))
        if abs(scale - 1.0) > 0.05:
            new_w = max(64, int(w * scale))
            new_h = max(64, int(h * scale))
            img_bgra = cv2.resize(img_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img_bgra

    def _ensure_builtin_starters(self) -> None:
        """Pre-create high-precision starter masks (Spider-Man, Iron Man, Batman, Cyber) if not cached."""
        starters = ["spiderman", "iron man", "batman", "joker"]
        for name in starters:
            slug = slugify(name)
            folder = AVATARS_DIR / slug
            if not (folder / "texture.png").exists():
                mask_img = self._create_procedural_mask(name)
                pts = np.zeros_like(CANONICAL_UV_ARRAY, dtype=np.float32)
                h, w = mask_img.shape[:2]
                pts[:, 0] = CANONICAL_UV_ARRAY[:, 0] * w
                pts[:, 1] = CANONICAL_UV_ARRAY[:, 1] * h
                asset = AvatarAsset(
                    name=name.title(),
                    slug=slug,
                    texture_bgra=mask_img,
                    source_points=pts,
                    is_canonical_mask=True,
                )
                self._save_to_disk(asset)

    def _create_procedural_mask(self, character_name: str) -> np.ndarray:
        """
        Synthesize a crisp 512x512 UV-aligned character mask aligned to CANONICAL_UV_ARRAY
        so presets like Spider-Man, Iron Man, Batman, and Joker work 100% offline with zero latency.
        """
        size = 512
        img = np.zeros((size, size, 4), dtype=np.uint8)
        lower = character_name.lower()

        # Outer face oval polygon in 512x512 UV space
        oval_uv = (CANONICAL_UV_ARRAY[:36] * size).astype(np.int32)
        hull = cv2.convexHull(oval_uv)

        if "spider" in lower:
            # Classic Crimson Spider-Man Mask with Webbing & White Lenses
            cv2.fillConvexPoly(img, hull, (30, 25, 215, 255))  # Rich Spider-Man Red (BGR)

            # Shading along jaw/temples
            shade = np.zeros_like(img)
            cv2.polylines(shade, [hull], True, (15, 10, 120, 255), 28, cv2.LINE_AA)
            img = cv2.addWeighted(img, 0.85, shade, 0.35, 0)

            # Radial web lines from nose bridge (0.50, 0.45)
            center = (int(0.50 * size), int(0.44 * size))
            for angle_deg in range(0, 360, 22):
                rad = np.deg2rad(angle_deg)
                end_x = int(center[0] + np.cos(rad) * size * 0.55)
                end_y = int(center[1] + np.sin(rad) * size * 0.55)
                cv2.line(img, center, (end_x, end_y), (20, 20, 25, 255), 2, cv2.LINE_AA)

            # Concentric web rings
            for r in range(36, 280, 38):
                cv2.circle(img, center, r, (20, 20, 25, 255), 2, cv2.LINE_AA)

            # Iconic Spider-Man Bold Black Eye Frames & White Lenses
            left_lens = np.array(
                [[85, 170], [135, 148], [208, 192], [182, 226], [112, 212]],
                dtype=np.int32,
            )
            right_lens = np.array(
                [[size - 85, 170], [size - 135, 148], [size - 208, 192], [size - 182, 226], [size - 112, 212]],
                dtype=np.int32,
            )
            cv2.fillPoly(img, [left_lens, right_lens], (245, 248, 255, 255))
            cv2.polylines(img, [left_lens, right_lens], True, (12, 12, 15, 255), 12, cv2.LINE_AA)

        elif "iron" in lower:
            # Iron Man Mark LXXXV Red & Gold Helmet
            cv2.fillConvexPoly(img, hull, (25, 30, 185, 255))  # Metallic Crimson
            faceplate = np.array(
                [
                    [95, 95], [200, 125], [312, 125], [417, 95],
                    [445, 250], [395, 420], [320, 465], [192, 465],
                    [117, 420], [67, 250],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(img, [faceplate], (45, 190, 245, 255))  # Gold Alloy (BGR)
            cv2.polylines(img, [faceplate], True, (20, 25, 90, 255), 5, cv2.LINE_AA)

            # Glowing Cyan Arc-Reactor Eyes
            l_eye = np.array([[105, 190], [200, 195], [190, 215], [115, 210]], dtype=np.int32)
            r_eye = np.array([[407, 190], [312, 195], [322, 215], [397, 210]], dtype=np.int32)
            cv2.fillPoly(img, [l_eye, r_eye], (255, 245, 180, 255))
            cv2.polylines(img, [l_eye, r_eye], True, (255, 200, 0, 255), 6, cv2.LINE_AA)

            # Jaw slit line
            jaw_line = np.array([[155, 395], [210, 380], [302, 380], [357, 395]], dtype=np.int32)
            cv2.polylines(img, [jaw_line], False, (20, 20, 30, 255), 6, cv2.LINE_AA)

        elif "bat" in lower:
            # Dark Knight Tactical Cowl (Covers forehead, eyes, nose; exposes lower jaw/mouth)
            cv2.fillConvexPoly(img, hull, (28, 28, 32, 255))
            # Cut out lower mouth area for classic Batman cowl look
            mouth_cut = np.array(
                [[140, 345], [256, 325], [372, 345], [345, 485], [256, 505], [167, 485]],
                dtype=np.int32,
            )
            cv2.fillPoly(img, [mouth_cut], (150, 175, 210, 255))
            # Glowing white tactical eyes
            l_eye = np.array([[105, 188], [195, 196], [180, 216], [115, 210]], dtype=np.int32)
            r_eye = np.array([[407, 188], [317, 196], [332, 216], [397, 210]], dtype=np.int32)
            cv2.fillPoly(img, [l_eye, r_eye], (250, 250, 255, 255))
            # Brow ridges
            cv2.line(img, (95, 170), (225, 190), (65, 65, 75, 255), 5, cv2.LINE_AA)
            cv2.line(img, (417, 170), (287, 190), (65, 65, 75, 255), 5, cv2.LINE_AA)

        elif "joker" in lower:
            # Classic Clown Prince Face Paint
            cv2.fillConvexPoly(img, hull, (235, 238, 242, 255))
            # Blue diamond eye paint
            l_diamond = np.array([[150, 130], [210, 195], [150, 265], [95, 195]], dtype=np.int32)
            r_diamond = np.array([[362, 130], [417, 195], [362, 265], [302, 195]], dtype=np.int32)
            cv2.fillPoly(img, [l_diamond, r_diamond], (190, 95, 20, 255))
            # Red nose tip
            cv2.circle(img, (256, 305), 32, (25, 20, 210, 255), -1, cv2.LINE_AA)
            # Glasgow smile
            smile = np.array(
                [[105, 345], [175, 390], [256, 405], [337, 390], [407, 345], [340, 435], [256, 450], [172, 435]],
                dtype=np.int32,
            )
            cv2.fillPoly(img, [smile], (20, 15, 215, 255))

        else:
            # Deterministic Cyber-Hero Mask based on character name hash
            seed = sum(ord(c) * (i + 1) for i, c in enumerate(lower))
            hue = seed % 180
            hsv_base = np.uint8([[[hue, 210, 210]]])
            bgr_base = cv2.cvtColor(hsv_base, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.fillConvexPoly(img, hull, (*bgr_base, 255))

            # Visor & Cyber circuitry
            visor = np.array([[85, 165], [256, 190], [427, 165], [405, 230], [256, 245], [107, 230]], dtype=np.int32)
            cv2.fillPoly(img, [visor], (255, 255, 0, 255))
            cv2.polylines(img, [visor], True, (20, 20, 20, 255), 6, cv2.LINE_AA)
            cv2.putText(
                img,
                character_name.upper()[:10],
                (155, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return img
