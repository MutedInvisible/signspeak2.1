# SignSpeak

Train your own sign language on a webcam, then get live translations in a desktop UI.

It does **not** ship a giant pre-trained ASL dictionary. You teach it *your* signs (letters, words, or homemade gestures). MediaPipe tracks your hands; a small classifier maps those poses (and motions) to the labels you captured.

## New features

1. **Bidirectional**
   - Sign → English (live camera translation)
   - English → Text (type English, show it large + speak it)

2. **Motion hand signals**
   - Switch to **Motion** mode for signs that move (e.g. ASL “cat”)
   - Captures a short sequence and learns the movement pattern
   - Works with 1 or 2 hands

3. **Improved Text-to-Speech**
   - Louder (volume maxed)
   - Slightly slower & clearer
   - Small intentional delay before speaking
   - Tries to pick a better system voice

4. **One-hand / Two-hand mode**
   - Toggle **1 Hand** or **2 Hands** before capturing or translating
   - Two-hand mode is required for signs that use both hands

5. **Custom UI**
   - Settings tab: change background, card, accent, text colors (hex)
   - Adjust font size
   - Optional background image path (image sits behind; text stays readable)

## Run

Use Python 3.12 (MediaPipe is picky about newer versions):

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Allow camera access if Windows asks.

## How to use

### Train a static sign
1. Open **Train** → leave **Static** selected.
2. Choose **1 Hand** or **2 Hands**.
3. Type a meaning (`hello`, `a`, `thanks`).
4. Hold the pose and press **Hold to capture**.
5. Capture a few samples, then **Train model**.
6. Sign in front of the camera. Hold ~0.7 s to add the word to the sentence.

### Train a motion sign
1. Switch to **Motion**.
2. Type the meaning (e.g. `cat`).
3. Press and hold **Hold to capture motion** while you perform the full movement.
4. Capture several good takes, then **Train model** (while still in Motion mode).
5. Live: perform the same motion; the app aggregates the recent frames and predicts.

### English → Text
- Open the **English → Text** tab.
- Type any English sentence → **Show** (big readable text) or **Speak it**.

### Customize look
- Open **Settings**.
- Change hex colors, font size, or set a path to a background image.
- Click **Apply theme**.

Samples live in `data/` so you can close the app and keep training later.

## About

SPEAKAUP
