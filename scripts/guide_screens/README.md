# Guide screenshots

The pictures in the in-app guide (the **?** button) and the welcome card, in light
and dark. When the layout changes, retake them:

```bash
cd scripts/guide_screens
npm install && npx playwright install chromium   # once
node capture.mjs                                  # local server on :7860, builds 1TUP
python build_images.py                            # needs Pillow
```

`capture.mjs --url https://pdb2print.org --id 1TUP` shoots the live site instead.

`build_images.py` writes `frontend/img/guide/*.webp` and redraws the numbered boxes
in `frontend/index.html` from where each control was in the capture. The numbers
match the lists under each picture in the guide; if you add or remove a box, change
the list in the same commit.
