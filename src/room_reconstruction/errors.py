"""Domain-specific failures with actionable user-facing messages."""


class ReconstructionError(RuntimeError):
    """Base class for expected reconstruction failures."""


class InputValidationError(ReconstructionError):
    """The input video does not satisfy pipeline requirements."""


class CommandUnavailableError(ReconstructionError):
    """A required external executable is unavailable."""


class ExternalCommandError(ReconstructionError):
    """An external command returned a non-zero exit code."""


class RegistrationError(ReconstructionError):
    """Camera registration is too weak to train a useful scene."""

