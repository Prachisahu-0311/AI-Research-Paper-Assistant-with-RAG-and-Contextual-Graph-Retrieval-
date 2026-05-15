"""Run before deploying — checks all required files and data exist."""
from pathlib import Path

checks = [
    ("data/chroma_db", "Vector store — run scripts/ingest_all.py first"),
    ("data/graph/raw_triples.json", "Graph triples — must exist"),
    ("src/api.py", "FastAPI app"),
    ("render.yaml", "Render config"),
    (".env", "Environment variables"),
]

all_ok = True
for path, description in checks:
    exists = Path(path).exists()
    status = "OK" if exists else "MISSING"
    print(f"[{status}] {path} — {description}")
    if not exists:
        all_ok = False

if all_ok:
    print("\nAll checks passed. Ready to deploy.")
    print("Next steps:")
    print("1. Push to GitHub")
    print("2. Connect repo to Render")
    print("3. Add GROQ_API_KEY environment variable in Render dashboard")
    print("4. Update API_URL in frontend/index.html to your Render URL")
    print("5. Deploy frontend/index.html to Vercel or GitHub Pages")
else:
    print("\nFix missing items before deploying.")
