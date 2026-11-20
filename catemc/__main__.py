"""Allow ``python -m catemc`` as an alternative to the ``catemc`` command."""
import sys

from catemc.cli import main

if __name__ == '__main__':
    sys.exit(main())
