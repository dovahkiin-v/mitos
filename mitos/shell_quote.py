"""Shell quoting for the values a printed ``mitos …`` recipe interpolates.

A recipe is a string the reader is told to RUN (ADR
``recipe-interpolated-project-identity-is-shell-quoted-not-raw``), so the value it
carries — a registered name, a workspace path, a slug — must survive a POSIX shell and
then mitos's own argparse. ``repr`` was the shipped idiom and is right for ordinary
values, but it is a Python literal: it breaks on a value holding both quote kinds or a
backslash, and a dash-led value comes out of the shell as a word argparse reads as an
option. This leaf is the one spelling for both halves.

Import-free on purpose: ``recall`` and ``lexical`` render recipes too,
and ``recall`` sits in the check family's fenced import closure, so whatever it imports
must be a pure leaf.
"""


def shell_arg(value: str) -> str:
    """Quotes one value for a POSIX shell, single-quoted.

    An ordinary value renders exactly as ``repr`` did (``'demo'``), so the shipped
    recipe form is unchanged; a value holding ``'`` closes, escapes and reopens the
    quote (``'it'"'"'s'``). A value carrying a non-printable character falls back to
    ``repr``'s escaped form: no recipe can carry it usefully, and printing it raw would
    write a newline or an ESC into the reader's terminal (neutralizing those everywhere
    is entry-018's sweep, still deferred).

    Args:
        value: The value to quote.

    Returns:
        The quoted value.
    """
    if not value.isprintable():
        return repr(value)
    return "'" + value.replace("'", "'\"'\"'") + "'"


def shell_flag(flag: str, value: str) -> str:
    """Renders ``flag 'value'``, or ``flag='-value'`` when the value looks like a flag.

    argparse reads a separate dash-led word as an option, so ``-p '-x'`` fails after
    the shell strips the quotes; the ``=`` form binds the value to its flag. Both
    ``-p=`` and ``--long=`` parse.

    Args:
        flag: The option, e.g. ``-p`` or ``--reconcile-entry``.
        value: Its value.

    Returns:
        The fragment, ready to splice into a recipe.
    """
    quoted = shell_arg(value)
    return f"{flag}={quoted}" if value.startswith("-") else f"{flag} {quoted}"
