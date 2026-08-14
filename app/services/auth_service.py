"""
Authentication service for CookHero.

Provides user registration, password hashing, and JWT token generation.
Includes security features: account lockout after failed attempts.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import bcrypt
from jose import jwt
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from redis.asyncio import Redis

from app.config import settings
from app.database.models import UserModel
from app.database.session import get_session_context

logger = logging.getLogger(__name__)


class AuthService:
    """Service for user authentication and registration with security features."""

    def __init__(self, secret_key: str | None = None):
        self.secret_key = secret_key or settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.max_failed_attempts = settings.LOGIN_MAX_FAILED_ATTEMPTS
        self.lockout_minutes = settings.LOGIN_LOCKOUT_MINUTES
        self._redis: Optional[Redis] = None

    def set_redis(self, redis_client: Redis) -> None:
        """Set Redis client for login attempt tracking."""
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Login attempt tracking (account lockout protection)
    # ------------------------------------------------------------------
    async def _get_failed_attempts_key(self, username: str) -> str:
        """Generate Redis key for tracking failed login attempts."""
        return f"auth:failed_attempts:{username}"

    async def _get_lockout_key(self, username: str) -> str:
        """Generate Redis key for account lockout."""
        return f"auth:lockout:{username}"

    async def is_account_locked(self, username: str) -> Tuple[bool, int]:
        """
        Check if account is locked due to too many failed attempts.

        Returns:
            Tuple of (is_locked, remaining_seconds)
        """
        if not self._redis:
            return False, 0

        try:
            lockout_key = await self._get_lockout_key(username)
            ttl = await self._redis.ttl(lockout_key)
        except Exception as e:
            # Redis 不可用：降级为不锁定（记录告警，避免登录路径崩溃）
            logger.warning("Redis unavailable for lockout check, degraded: %s", e)
            return False, 0

        if ttl > 0:
            return True, ttl

        return False, 0

    async def record_failed_attempt(self, username: str) -> Tuple[int, bool]:
        """
        Record a failed login attempt.

        Returns:
            Tuple of (current_attempts, is_now_locked)
        """
        if not self._redis:
            return 0, False

        try:
            failed_key = await self._get_failed_attempts_key(username)
            lockout_key = await self._get_lockout_key(username)

            # Increment failed attempts
            attempts = await self._redis.incr(failed_key)

            # Set expiry on the counter (reset after lockout period)
            await self._redis.expire(failed_key, self.lockout_minutes * 60)
        except Exception as e:
            logger.warning("Redis unavailable for failed-attempt tracking, degraded: %s", e)
            return 0, False

        # Check if we need to lock the account
        if attempts >= self.max_failed_attempts:
            # Lock the account
            try:
                await self._redis.setex(
                    lockout_key,
                    self.lockout_minutes * 60,
                    "locked"
                )
            except Exception as e:
                logger.warning("Redis unavailable for lockout, degraded: %s", e)
                return attempts, False
            logger.warning(f"Account locked: {username} after {attempts} failed attempts")
            return attempts, True

        return attempts, False

    async def clear_failed_attempts(self, username: str) -> None:
        """Clear failed login attempts after successful login."""
        if not self._redis:
            return

        try:
            failed_key = await self._get_failed_attempts_key(username)
            lockout_key = await self._get_lockout_key(username)

            await self._redis.delete(failed_key, lockout_key)
        except Exception as e:
            logger.warning("Redis unavailable for attempt cleanup, degraded: %s", e)

    # ------------------------------------------------------------------
    # User retrieval helpers
    # ------------------------------------------------------------------
    async def get_user_by_username(self, username: str) -> Optional[UserModel]:
        async with get_session_context() as session:
            stmt = select(UserModel).where(UserModel.username == username)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Registration & authentication
    # ------------------------------------------------------------------
    async def register_user(self, username: str, password: str) -> UserModel:
        existing = await self.get_user_by_username(username)
        if existing:
            raise ValueError("Username already exists")

        password_hash = self._hash_password(password)
        user = UserModel(username=username, password_hash=password_hash)

        async with get_session_context() as session:
            session.add(user)
            try:
                await session.flush()
            except IntegrityError as exc:
                logger.warning("Integrity error on user register: %s", exc)
                raise ValueError("Username already exists")

        logger.info("Created user %s", username)
        return user

    async def authenticate_user(self, username: str, password: str) -> Optional[UserModel]:
        user = await self.get_user_by_username(username)
        if not user:
            return None
        if not self._verify_password(password, user.password_hash):
            return None
        return user

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------
    def create_access_token(self, user: UserModel) -> str:
        expire = datetime.now() + timedelta(minutes=self.expire_minutes)
        payload = {"sub": user.username, "uid": str(user.id), "exp": expire}
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return {"username": payload.get("sub"), "user_id": payload.get("uid")}
        except Exception as exc:  # broad catch to avoid import of ExpiredSignatureError etc
            logger.warning("Failed to decode token: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Password helpers
    # ------------------------------------------------------------------
    def _hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def _verify_password(self, password: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except ValueError:
            return False


auth_service = AuthService()
