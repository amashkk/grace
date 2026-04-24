"""Wrapper to run GRACE with full error visibility."""
import sys
import traceback

if __name__ == "__main__":
    try:
        from main import main
        main()
    except SystemExit:
        pass
    except Exception:
        traceback.print_exc()
        sys.exit(1)
