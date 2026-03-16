"""Authentication service providing password hashing, JWT management, and user operations.

This module implements the core authentication logic for the application,
including bcrypt password hashing, JWT token creation/validation, user CRUD
operations, and FastAPI dependency injection functions for route protection.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import Settings
from app.database import get_db
from app.models.user import User
from app.schemas.user import TokenData

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

settings = Settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a bcrypt hash.

    Args:
        plain_password: The plain-text password to check.
        hashed_password: The bcrypt hash to compare against.

    Returns:
        True if the password matches the hash, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a plain-text password using bcrypt.

    Args:
        password: The plain-text password to hash.

    Returns:
        The resulting bcrypt hash string.
    """
    return pwd_context.hash(password)


# ---------------------------------------------------------------------------
# JWT token management
# ---------------------------------------------------------------------------


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a short-lived JWT access token.

    Args:
        data: Payload data to encode in the token (must include ``sub``).
        expires_delta: Optional custom expiration duration. Defaults to the
            value configured in ``Settings.ACCESS_TOKEN_EXPIRE_MINUTES``.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a long-lived JWT refresh token.

    Args:
        data: Payload data to encode in the token (must include ``sub``).

    Returns:
        Encoded JWT string with a longer expiration window.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token.

    Args:
        token: The encoded JWT string.

    Returns:
        The decoded token payload as a dictionary.

    Raises:
        HTTPException: 401 if the token is expired, malformed, or otherwise
            fails validation.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ---------------------------------------------------------------------------
# User CRUD operations
# ---------------------------------------------------------------------------


async def get_user_by_email(db: Session, email: str) -> User | None:
    """Look up a user by their email address.

    Args:
        db: SQLAlchemy database session.
        email: The email address to search for.

    Returns:
        The ``User`` instance if found, or ``None``.
    """
    return db.query(User).filter(User.email == email).first()


async def create_user(db: Session, email: str, password: str) -> User:
    """Create a new user with a hashed password.

    Args:
        db: SQLAlchemy database session.
        email: The user's email address.
        password: The plain-text password (will be hashed before storage).

    Returns:
        The newly created ``User`` instance.
    """
    hashed_password = get_password_hash(password)
    user = User(
        email=email,
        hashed_password=hashed_password,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


async def authenticate_user(db: Session, email: str, password: str) -> User | None:
    """Authenticate a user by email and password.

    If authentication succeeds the user's ``last_login`` timestamp is updated.

    Args:
        db: SQLAlchemy database session.
        email: The email address provided at login.
        password: The plain-text password provided at login.

    Returns:
        The authenticated ``User`` instance, or ``None`` if authentication
        fails (wrong email or wrong password).
    """
    user = await get_user_by_email(db, email)
    if user is None:
        return None
    if not verify_password(password, user.hashed_password):
        return None

    # Update last_login timestamp
    user.last_login = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency that extracts and validates the current user from a JWT.

    Args:
        token: Bearer token extracted from the ``Authorization`` header.
        db: SQLAlchemy database session.

    Returns:
        The authenticated ``User`` instance.

    Raises:
        HTTPException: 401 if the token is invalid or the user does not exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_token(token)
    email: Optional[str] = payload.get("sub")
    if email is None:
        raise credentials_exception

    token_type = payload.get("type")
    if token_type != "access":
        raise credentials_exception

    token_data = TokenData(email=email)
    user = await get_user_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """FastAPI dependency ensuring the current user's account is active.

    Args:
        current_user: The user resolved by ``get_current_user``.

    Returns:
        The same ``User`` instance if the account is active.

    Raises:
        HTTPException: 401 if the user's account has been deactivated.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user account",
        )
    return current_user


async def get_optional_user(
    token: Optional[str] = Depends(OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """FastAPI dependency that returns the current user or ``None``.

    Unlike ``get_current_user``, this dependency does **not** raise an
    exception when no valid token is provided. It is intended for endpoints
    that behave differently for authenticated vs. anonymous users.

    Args:
        token: Optional bearer token from the ``Authorization`` header.
        db: SQLAlchemy database session.

    Returns:
        The authenticated ``User`` if a valid token is present, otherwise ``None``.
    """
    if token is None:
        return None

    try:
        payload = decode_token(token)
        email: Optional[str] = payload.get("sub")
        if email is None:
            return None

        token_type = payload.get("type")
        if token_type != "access":
            return None

        user = await get_user_by_email(db, email=email)
        return user
    except HTTPException:
        return None
