"""Domain-specific failures with actionable user-facing messages."""


class ReconstructionError(RuntimeError):
    """Base class for expected reconstruction failures."""


class InputValidationError(ReconstructionError):
    """The input video does not satisfy pipeline requirements."""


class FrameQualityError(ReconstructionError):
    """Extracted frames could not be scored or filtered."""


class ConfigurationError(ReconstructionError):
    """A pipeline configuration file is missing or invalid."""


class CommandUnavailableError(ReconstructionError):
    """A required external executable is unavailable."""


class ExternalCommandError(ReconstructionError):
    """An external command returned a non-zero exit code."""


class RegistrationError(ReconstructionError):
    """Camera registration is too weak to train a useful scene."""


class OutputExistsError(ReconstructionError):
    """An output directory contains artifacts that require an explicit resume."""


class ExportError(ReconstructionError):
    """A completed reconstruction could not be exported."""
