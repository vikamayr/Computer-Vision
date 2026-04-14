# Detector Skill (AI Edge Gallery)

This repo contains a custom Agent Skill for Google AI Edge Gallery:

- Skill folder: `detector-skill/`
- Entry files:
  - `detector-skill/SKILL.md`
  - `detector-skill/scripts/index.html`
  - `detector-skill/assets/webview.html`

## 1) Push to GitHub

From this folder:

1. Initialize git (if needed), commit, and push to your GitHub repository.
2. Keep `detector-skill/` at repository root or update URLs accordingly.

## 2) Enable GitHub Pages

1. In your GitHub repo settings, enable **Pages** from the main branch.
2. Wait for deployment URL (example):
	- `https://<your-user>.github.io/<your-repo>/`

## 3) Skill URL to load in AI Edge Gallery

Use the URL to the skill folder (without `SKILL.md` suffix):

- `https://<your-user>.github.io/<your-repo>/detector-skill`

Quick check in browser:

- `https://<your-user>.github.io/<your-repo>/detector-skill/SKILL.md`

If this opens as raw markdown text, the URL is valid for Gallery import.

## 4) Load into AI Edge Gallery

1. Open Agent Skills in the app.
2. Tap **+** → **Load skill from URL**.
3. Paste the skill folder URL above.
4. Select Gemma 4 and test prompts like:
	- "Detect humans and animals from my camera"

## 5) Notes

- Real-time box detection is performed in webview by MediaPipe Tasks.
- Gemma is used for agent/tool orchestration and optional reasoning over exported JSON.
