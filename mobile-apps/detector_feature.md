# Feature Spec: Camera Bounding Box Detection Skill (Humans + Animals)

## 1) Summary
Build a new Google AI Edge Gallery JS skill that opens a live phone camera view and returns/uses structured bounding-box detections for:
- humans
- animals

Primary target is Gemma 4 multimodal reasoning for detection text output (boxes + labels + confidence), with UI overlay handled by skill code.

## 2) Why this fits AI Edge Gallery
Based on the AI Edge Gallery skills docs:
- Skill folder must contain `SKILL.md`.
- JS skills execute through `scripts/index.html` and `window['ai_edge_gallery_get_result']`.
- Skills can return a `webview` payload for interactive UI.
- Camera-enabled examples exist (e.g., `text-spinner` webview pattern).

This feature should therefore be implemented as a JS skill that returns an interactive webview where camera + visualization runs.

## 3) Scope (Phase 1)
### In scope
- Live camera stream in webview.
- Continuous detection loop for humans/animals.
- Bounding boxes drawn on top of video.
- Structured detection JSON output format defined and used consistently.
- Basic controls: start/stop detection, FPS throttle, confidence threshold.
- Mirror handling for front camera.

### Out of scope (for now)
- Full COCO/all-object catalog.
- Tracking IDs across long sequences.
- Segmentation masks.
- Cloud dependency requirement.

## 4) User stories
1. As a user, I can ask for live detection and see boxes over people/animals in camera preview.
2. As a user, I can switch camera facing mode and still get correct box coordinates.
3. As a user, I can read a compact text summary of current detections.
4. As a developer, I can reuse a stable JSON schema for detections.

## 5) Functional requirements

### FR-1 Skill invocation
The skill must be invokable through `run_js` with a minimal `data` JSON payload.

Proposed input schema:
```json
{
  "mode": "live_detect",
  "targets": ["human", "animal"],
  "min_confidence": 0.45,
  "max_fps": 5
}
```

### FR-2 Webview startup
`scripts/index.html` must return a `webview.url` pointing to interactive UI (e.g. `webview.html?...`).

### FR-3 Camera pipeline
Webview must:
- request camera permission,
- render video full-screen,
- run frame sampling loop (`max_fps` throttled),
- preserve correct coordinates for portrait/landscape.

### FR-4 Detection contract (canonical)
Each frame detection payload should conform to:
```json
{
  "timestamp_ms": 1710000000000,
  "image": {
    "width": 1280,
    "height": 720
  },
  "detections": [
    {
      "label": "human",
      "confidence": 0.93,
      "bbox": {
        "x": 0.12,
        "y": 0.18,
        "w": 0.31,
        "h": 0.64
      }
    },
    {
      "label": "dog",
      "confidence": 0.88,
      "bbox": {
        "x": 0.55,
        "y": 0.40,
        "w": 0.22,
        "h": 0.30
      }
    }
  ]
}
```
Notes:
- Bounding box coordinates are normalized to [0,1].
- `label` must map to target classes (`human` or animal subtype such as `dog`, `cat`, `bird`, etc.).

### FR-5 Visualization
Overlay must draw:
- rectangle,
- label text,
- confidence value.

### FR-6 Filtering
- Only human/animal classes are displayed in Phase 1.
- Detections below `min_confidence` are discarded.

### FR-7 Result text
Skill should provide a short textual summary (for chat/history), e.g.:
- "Detected 2 humans, 1 dog."

## 6) Gemma 4 multimodal requirement
Target behavior: Gemma 4 interprets frame image input and outputs strict JSON bounding boxes.

Because AI Edge Gallery JS skills run in webview context, we should implement a **model adapter layer** with this interface:
- `detect(frameBitmapOrBase64) -> DetectionPayload`

Adapter responsibilities:
1. Prompt Gemma 4 with strict JSON schema request.
2. Parse/validate JSON safely.
3. Normalize labels to Phase-1 taxonomy.
4. Return canonical detection payload.

### Important implementation note
If direct frame-to-Gemma calls are not exposed in the skill runtime path, keep the same detection schema and use a local vision detector adapter first, then swap adapter to Gemma later without changing UI contract.

## 7) UX requirements
- Show status badge: `initializing`, `running`, `permission_denied`, `error`.
- Keep overlay responsive at low/medium-end device performance.
- Include simple legend/colors:
  - human: cyan
  - animal: lime

## 8) Error handling
- Camera permission denied -> show actionable message.
- Model parse failure -> skip frame and continue loop.
- No detections -> render "No humans/animals detected" state.

## 9) Performance targets (Phase 1)
- End-to-end refresh >= 3 FPS on mid-range phone.
- Inference loop must not block UI thread.
- Frame sampling + rendering jitter should remain visually stable.

## 10) Security & privacy
- All processing should remain on-device when possible.
- No persistent storage of raw frames by default.
- No hidden network upload of camera frames unless user explicitly enables it.

## 11) Skill packaging requirements
Skill repo structure to keep:
- `detector-skill/SKILL.md`
- `detector-skill/scripts/index.html`
- (next step) `detector-skill/assets/webview.html`

Optional for hosting via GitHub Pages:
- Add `.nojekyll` at repo root to ensure raw `SKILL.md` is served correctly.

## 12) Acceptance criteria
1. User can trigger skill from chat and open preview card.
2. Camera opens in webview and overlay appears.
3. Humans and at least common animals are boxed with labels/confidence.
4. Box coordinates follow normalized schema and map correctly on screen.
5. Skill remains functional when hosted from URL and loaded in AI Edge Gallery.

## 13) Implementation plan (next)
1. Update `SKILL.md` with strict `run_js` instructions and data schema.
2. Implement `scripts/index.html` to return webview + config.
3. Add `assets/webview.html` with camera, inference adapter, and canvas overlay.
4. Add schema validator and class filter for human/animal.
5. Add basic manual test prompts and edge-case tests.
