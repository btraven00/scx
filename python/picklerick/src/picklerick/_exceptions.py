"""Exceptions for the picklerick Python package."""


class PickleRickError(Exception):
    """Base exception for all picklerick errors."""


class ScxNotFoundError(PickleRickError):
    """Raised when the ``scx`` executable cannot be located."""


class ScxCommandError(PickleRickError):
    """Raised when an ``scx`` subprocess exits with a non-zero status."""

    def __init__(self, message: str, command: list[str] | None = None, stderr: str | None = None):
        super().__init__(message)
        self.command = command
        self.stderr = stderr