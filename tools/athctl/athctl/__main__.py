"""Entry point for ``python -m athctl``.

Delegates to the Click group declared in ``athctl.cli``.
"""

from athctl.cli import main

if __name__ == "__main__":
    main()
