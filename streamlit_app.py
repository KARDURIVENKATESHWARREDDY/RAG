"""Streamlit Cloud entrypoint.

Streamlit Community Cloud looks for `streamlit_app.py` by default.
This file simply runs the main app from `app.py`.
"""

from app import main


if __name__ == "__main__":
    main()

