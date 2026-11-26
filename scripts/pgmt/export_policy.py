from pathlib import Path

from instinctlab.tasks.shadowing.pgmt.supermimic.deploy_export import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2] / "tmp" / "export")
