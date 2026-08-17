"""
Sensitive data sanitization for CookHero.

Provides logging filters and utilities to prevent
sensitive information leakage in logs and responses.
"""

import logging
import re
from typing import Any, Dict, Optional, Set


class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that masks sensitive data in log messages.

    Masks common sensitive patterns like passwords, tokens, API keys.
    """

    # Patterns to match and mask (key patterns)
    SENSITIVE_KEYS = {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "auth",
        "credential",
        "private_key",
        "access_token",
        "refresh_token",
        "jwt",
        "bearer",
    }

    # Regex patterns for inline sensitive data
    SENSITIVE_PATTERNS = [
        # API keys (various formats)
        (re.compile(r'(sk-[a-zA-Z0-9]{20,})', re.IGNORECASE), r'sk-***MASKED***'),
        (re.compile(r'(api[_-]?key["\s:=]+)["\']?([a-zA-Z0-9_-]{16,})["\']?', re.IGNORECASE), r'\1***MASKED***'),

        # Bearer tokens
        (re.compile(r'(bearer\s+)([a-zA-Z0-9._-]{20,})', re.IGNORECASE), r'\1***MASKED***'),

        # JWT tokens (header.payload.signature)
        (re.compile(r'(eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)', re.IGNORECASE), r'***JWT_MASKED***'),

        # Password in URLs
        (re.compile(r'(://[^:]+:)([^@]+)(@)', re.IGNORECASE), r'\1***@'),

        # Email patterns (partial mask)
        (re.compile(r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', re.IGNORECASE),
         lambda m: f"{m.group(1)[:2]}***@{m.group(2)}"),
    ]

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._sensitive_key_pattern = re.compile(
            r'(["\']?)(' + '|'.join(self.SENSITIVE_KEYS) + r')(["\']?\s*[:=]\s*)["\']?([^"\'\s,}\]]+)["\']?',
            re.IGNORECASE
        )

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and sanitize log record."""
        # Sanitize the message
        if isinstance(record.msg, str):
            record.msg = self._sanitize_string(record.msg)

        # Sanitize args if present
        if record.args:
            if isinstance(record.args, dict):
                record.args = self._sanitize_dict(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._sanitize_value(arg) for arg in record.args
                )

        return True

    def _sanitize_string(self, text: str) -> str:
        """Sanitize sensitive data in a string."""
        if not text:
            return text

        # Mask key-value pairs with sensitive keys
        text = self._sensitive_key_pattern.sub(
            r'\1\2\3"***MASKED***"', text
        )

        # Apply pattern-based masking
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            if callable(replacement):
                text = pattern.sub(replacement, text)
            else:
                text = pattern.sub(replacement, text)

        return text

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a single value."""
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, dict):
            return self._sanitize_dict(value)
        elif isinstance(value, (list, tuple)):
            return type(value)(self._sanitize_value(v) for v in value)
        return value

    def _sanitize_dict(self, data: Dict) -> Dict:
        """Sanitize sensitive keys in a dictionary."""
        result = {}
        for key, value in data.items():
            lower_key = str(key).lower()

            # Check if key is sensitive
            if any(sk in lower_key for sk in self.SENSITIVE_KEYS):
                result[key] = "***MASKED***"
            else:
                result[key] = self._sanitize_value(value)

        return result


class Sanitizer:
    """
    Utility class for sanitizing sensitive data.

    Use for sanitizing data before logging or returning in responses.
    """

    # Maximum content length in logs
    MAX_LOG_CONTENT_LENGTH = 500

    # Fields to completely mask
    MASKED_FIELDS: Set[str] = {
        "password",
        "password_hash",
        "secret_key",
        "api_key",
        "token",
        "access_token",
        "refresh_token",
    }

    # Fields to truncate
    TRUNCATED_FIELDS: Set[str] = {
        "content",
        "message",
        "context",
        "response",
        "body",
    }

    @classmethod
    def mask_sensitive_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask sensitive fields in a dictionary.

        Args:
            data: Dictionary to sanitize

        Returns:
            Sanitized copy of the dictionary
        """
        if not data:
            return data

        result = {}
        for key, value in data.items():
            lower_key = key.lower()

            if lower_key in cls.MASKED_FIELDS or any(
                mf in lower_key for mf in cls.MASKED_FIELDS
            ):
                result[key] = "***MASKED***"
            elif lower_key in cls.TRUNCATED_FIELDS:
                result[key] = cls.truncate(str(value))
            elif isinstance(value, dict):
                result[key] = cls.mask_sensitive_fields(value)
            elif isinstance(value, list):
                result[key] = [
                    cls.mask_sensitive_fields(v) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                result[key] = value

        return result

    @classmethod
    def truncate(cls, text: str, max_length: Optional[int] = None) -> str:
        """
        Truncate text to maximum length with indicator.

        Args:
            text: Text to truncate
            max_length: Maximum length (default: MAX_LOG_CONTENT_LENGTH)

        Returns:
            Truncated text
        """
        if not text:
            return text

        max_len = max_length or cls.MAX_LOG_CONTENT_LENGTH
        if len(text) <= max_len:
            return text

        return text[:max_len] + f"...[truncated, total {len(text)} chars]"

    @classmethod
    def mask_api_key(cls, api_key: str) -> str:
        """
        Mask API key showing only first and last few characters.

        Args:
            api_key: API key to mask

        Returns:
            Masked API key like "sk-abc...xyz"
        """
        if not api_key or len(api_key) < 8:
            return "***"

        # Show first 5 and last 3 characters
        return f"{api_key[:5]}...{api_key[-3:]}"

    @classmethod
    def safe_log_dict(cls, data: Dict[str, Any], max_length: int = 500) -> str:
        """
        Create a safe string representation of a dict for logging.

        Args:
            data: Dictionary to log
            max_length: Maximum total string length

        Returns:
            Safe string representation
        """
        sanitized = cls.mask_sensitive_fields(data)
        result = str(sanitized)

        if len(result) > max_length:
            return result[:max_length] + "..."

        return result


def setup_secure_logging() -> None:
    """
    Configure logging with sensitive data filtering.

    Call this during application startup to enable filtering
    on all log handlers.
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Add filter to all handlers
    sensitive_filter = SensitiveDataFilter()

    for handler in root_logger.handlers:
        handler.addFilter(sensitive_filter)

    # Also add to the root logger itself
    root_logger.addFilter(sensitive_filter)

    logging.info("Secure logging configured with sensitive data filtering")


# Global sanitizer instance
sanitizer = Sanitizer()
