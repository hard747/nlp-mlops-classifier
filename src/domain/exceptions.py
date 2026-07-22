class ClassificationError(Exception):
    """Raised when the underlying model fails to produce a prediction."""


class AuditPersistenceError(Exception):
    """Raised when an audit log entry cannot be persisted by the repository."""
