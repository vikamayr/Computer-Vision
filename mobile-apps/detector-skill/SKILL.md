---
name: detector-skill
description: Live camera detection for humans and animals with bounding boxes.
---

# Human and animal detector

This skill opens a live camera webview and overlays bounding boxes for humans and animals.

## Examples

- "Detect humans and animals from my camera"
- "Start live object detection and show boxes"
- "Use camera and detect people, dogs, and cats"
- "Start detector, then help me analyze snapshot JSON"

## Instructions

You MUST use the `run_js` tool with the following exact parameters:

- script name: `index.html`
- data: A JSON string with the following fields:
	- mode: String. Use `live_detect`.
	- targets: Array of strings. Allowed values: `human`, `animal`.
	- min_confidence: Number from 0 to 1. Suggested default: `0.45`.
	- max_fps: Number. Suggested range: `1` to `10`, default `5`.
	- facing_mode: String. Optional: `environment` or `user`.
	- gemma_assist: Boolean. Optional. If true, encourage exporting latest detection JSON for follow-up Gemma reasoning.

If the user does not provide values, use this default payload:

```json
{
	"mode": "live_detect",
	"targets": ["human", "animal"],
	"min_confidence": 0.45,
	"max_fps": 5,
	"facing_mode": "environment",
	"gemma_assist": true
}
```

Return the interactive camera preview card to the user. Do not claim final detections before the preview is opened.

If the user asks for deeper reasoning (scene understanding, suggestions, safety review, counting trends), ask them to tap **Copy JSON** in the preview and paste it back into chat, then analyze that JSON.