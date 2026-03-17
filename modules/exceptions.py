"""
Custom exceptions for the project.
"""


class AngelsBookError(Exception):
    """Base exception for the project."""

    pass


class DataNotFoundError(AngelsBookError):
    """Raised when a required data file is missing."""

    pass


class CheckpointNotFoundError(AngelsBookError):
    """Raised when a checkpoint file or directory is missing."""

    pass


class ConfigError(AngelsBookError):
    """Raised when configuration is invalid or missing."""

    pass


class TokenizerError(AngelsBookError):
    """Raised when tokenizer loading or encoding fails."""

    pass


class ModelLoadError(AngelsBookError):
    """Raised when model loading fails."""

    pass
