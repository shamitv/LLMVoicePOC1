This directory is intentionally ignored by Git.

Use this folder to store local model checkpoints, generated audio, and other large artifacts that should not be committed to the repository.

Notes:
- The repository `.gitignore` contains `backend/data/` so files placed here remain local only.
- To change the location the application uses, set the `DATA_DIR` environment variable (see `backend/.env.example`).
- If you need to share a model or artifact, add it to a release or provide a download script instead of committing it here.
