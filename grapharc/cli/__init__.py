"""The `grapharc` command line.

`grapharc.cli.main:main` is the console-script entry point. Nothing is
re-exported here: the commands import the components they drive at call time so
one missing optional package cannot take down `--help`.
"""
