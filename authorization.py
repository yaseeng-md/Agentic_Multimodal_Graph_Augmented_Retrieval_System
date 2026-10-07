"""
AMG Multimodal RAG - Authentication Foundation

This module is intentionally independent from api.py.

V1 responsibilities:
    1. Store users and organizations in SQLite.
    2. Hash and verify passwords.
    3. Create JWT bearer access tokens.
    4. Validate JWT access tokens.
    5. Resolve the authenticated user from the token.
    6. Expose a FastAPI-compatible authentication dependency.

Fixed V1 roles:
    - chief
    - manager
    - engineer
    - worker

Important:
    - A user has exactly one organization.
    - A user has exactly one role.
    - The client does NOT provide identity during protected requests.
    - The JWT establishes user identity through the `sub` claim.
    - Organization and current role are resolved from the database.
    - Detailed role permissions/authorization policies are intentionally
      deferred to a later phase.

Environment variables:
    AUTH_DB_PATH
        Default: ./data/auth.db

    JWT_SECRET_KEY
        Required. Use a long random secret in real deployments.

    JWT_ALGORITHM
        Default: HS256

    JWT_ISSUER
        Default: amg-rag-api

    JWT_AUDIENCE
        Default: amg-rag-client

    JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        Default: 60
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(
    os.getenv("DATA_DIR", str(BASE_DIR / "data"))
)

AUTH_DB_PATH = Path(
    os.getenv("AUTH_DB_PATH", str(DATA_DIR / "auth.db"))
)

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ISSUER = os.getenv("JWT_ISSUER", "amg-rag-api")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "amg-rag-client")

JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
)


# ============================================================================
# Fixed V1 roles
# ============================================================================

ROLE_CHIEF = "chief"
ROLE_MANAGER = "manager"
ROLE_ENGINEER = "engineer"
ROLE_WORKER = "worker"

ALLOWED_ROLES = frozenset(
    {
        ROLE_CHIEF,
        ROLE_MANAGER,
        ROLE_ENGINEER,
        ROLE_WORKER,
    }
)


# ============================================================================
# OAuth2 bearer scheme
# ============================================================================

# Used by FastAPI protected endpoints later.
#
# Example:
#
#     @app.get("/me")
#     def me(
#         current_user: AuthContext = Depends(get_current_user),
#     ):
#         ...
#
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/login",
)


# ============================================================================
# Authentication context
# ============================================================================

@dataclass(frozen=True)
class AuthContext:
    """
    Identity resolved from a validated access token + current DB state.

    This is what protected endpoints/services should consume rather than
    trusting user_id, organization, or role from the request body.
    """

    user_id: str
    name: str
    email: str
    organization_id: str
    organization_name: str
    role: str
    status: str

    def as_dict(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "organization_id": self.organization_id,
            "organization_name": self.organization_name,
            "role": self.role,
            "status": self.status,
        }


# ============================================================================
# General helpers
# ============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def new_user_id() -> str:
    return f"USR-{uuid.uuid4().hex[:12].upper()}"


def new_organization_id() -> str:
    return f"ORG-{uuid.uuid4().hex[:12].upper()}"


def normalise_email(email: str) -> str:
    value = email.strip().lower()

    if not value:
        raise ValueError("email must not be empty")

    return value


def normalise_name(name: str) -> str:
    value = " ".join(name.strip().split())

    if not value:
        raise ValueError("name must not be empty")

    return value


def normalise_organization_name(organization: str) -> str:
    value = " ".join(organization.strip().split())

    if not value:
        raise ValueError("organization must not be empty")

    return value


def normalise_role(role: str) -> str:
    value = role.strip().lower()

    if value not in ALLOWED_ROLES:
        allowed = ", ".join(sorted(ALLOWED_ROLES))
        raise ValueError(
            f"Invalid role '{role}'. Allowed roles: {allowed}"
        )

    return value


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "").strip()

    if not secret:
        raise RuntimeError(
            "JWT_SECRET_KEY is not configured. "
            "Set a strong random JWT_SECRET_KEY in the environment."
        )

    if len(secret) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY must be at least 32 characters long."
        )

    return secret


# ============================================================================
# Password hashing
# ============================================================================
#
# V1 uses Python's stdlib scrypt so the module does not add another runtime
# dependency just for password hashing.
#
# For a production identity service, replace this with a well-maintained
# Argon2id implementation.
# ============================================================================

_PASSWORD_PREFIX = "scrypt"
_PASSWORD_SALT_BYTES = 16
_PASSWORD_N = 2**14
_PASSWORD_R = 8
_PASSWORD_P = 1
_PASSWORD_DKLEN = 32


def hash_password(password: str) -> str:
    """Hash a password using salted scrypt."""
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")

    password_bytes = password.encode("utf-8")
    salt = secrets.token_bytes(_PASSWORD_SALT_BYTES)

    derived_key = hashlib.scrypt(
        password_bytes,
        salt=salt,
        n=_PASSWORD_N,
        r=_PASSWORD_R,
        p=_PASSWORD_P,
        dklen=_PASSWORD_DKLEN,
    )

    return (
        f"{_PASSWORD_PREFIX}$"
        f"{_PASSWORD_N}$"
        f"{_PASSWORD_R}$"
        f"{_PASSWORD_P}$"
        f"{salt.hex()}$"
        f"{derived_key.hex()}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against the stored scrypt hash."""
    if not password or not password_hash:
        return False

    try:
        prefix, n_raw, r_raw, p_raw, salt_hex, hash_hex = (
            password_hash.split("$")
        )

        if prefix != _PASSWORD_PREFIX:
            return False

        n = int(n_raw)
        r = int(r_raw)
        p = int(p_raw)

        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)

        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
        )

        return hmac.compare_digest(candidate, expected)

    except (ValueError, TypeError):
        return False


# ============================================================================
# SQLite authentication store
# ============================================================================

class AuthStore:
    """SQLite-backed storage for organizations and users."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,
        )

        conn.row_factory = sqlite3.Row

        # Better concurrency behavior for the current multi-threaded FastAPI
        # process. Each operation still gets its own connection.
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")

        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS organizations (
                    organization_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    name_normalized TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (
                        role IN (
                            'chief',
                            'manager',
                            'engineer',
                            'worker'
                        )
                    ),
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    FOREIGN KEY (organization_id)
                        REFERENCES organizations(organization_id)
                );

                CREATE INDEX IF NOT EXISTS idx_users_organization
                    ON users(organization_id);

                CREATE INDEX IF NOT EXISTS idx_users_status
                    ON users(status);
                """
            )

    def get_user_by_id(self, user_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT
                    u.*,
                    o.name AS organization_name,
                    o.status AS organization_status
                FROM users u
                JOIN organizations o
                    ON o.organization_id = u.organization_id
                WHERE u.user_id = ?
                """,
                (user_id,),
            ).fetchone()

    def get_user_by_email(self, email: str) -> sqlite3.Row | None:
        email = normalise_email(email)

        with self._connect() as conn:
            return conn.execute(
                """
                SELECT
                    u.*,
                    o.name AS organization_name,
                    o.status AS organization_status
                FROM users u
                JOIN organizations o
                    ON o.organization_id = u.organization_id
                WHERE u.email = ?
                """,
                (email,),
            ).fetchone()

    def get_or_create_organization(
        self,
        organization_name: str,
    ) -> sqlite3.Row:
        display_name = normalise_organization_name(organization_name)
        normalized = display_name.casefold()

        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT *
                FROM organizations
                WHERE name_normalized = ?
                """,
                (normalized,),
            ).fetchone()

            if existing is not None:
                return existing

            organization_id = new_organization_id()
            now = utc_now_iso()

            conn.execute(
                """
                INSERT INTO organizations (
                    organization_id,
                    name,
                    name_normalized,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (
                    organization_id,
                    display_name,
                    normalized,
                    now,
                    now,
                ),
            )

            return conn.execute(
                """
                SELECT *
                FROM organizations
                WHERE organization_id = ?
                """,
                (organization_id,),
            ).fetchone()

    def create_user(
        self,
        *,
        name: str,
        email: str,
        password_hash: str,
        organization_name: str,
        role: str,
    ) -> AuthContext:
        name = normalise_name(name)
        email = normalise_email(email)
        role = normalise_role(role)

        organization = self.get_or_create_organization(
            organization_name
        )

        organization_id = str(organization["organization_id"])

        user_id = new_user_id()
        now = utc_now_iso()

        with self._connect() as conn:
            # Explicitly check first so the public helper can return a clean
            # duplicate-email error before relying on the UNIQUE constraint.
            existing = conn.execute(
                """
                SELECT user_id
                FROM users
                WHERE email = ?
                """,
                (email,),
            ).fetchone()

            if existing is not None:
                raise ValueError(
                    f"A user with email '{email}' already exists."
                )

            conn.execute(
                """
                INSERT INTO users (
                    user_id,
                    name,
                    email,
                    password_hash,
                    organization_id,
                    role,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    user_id,
                    name,
                    email,
                    password_hash,
                    organization_id,
                    role,
                    now,
                    now,
                ),
            )

        row = self.get_user_by_id(user_id)

        if row is None:
            raise RuntimeError(
                f"User was created but could not be reloaded: {user_id}"
            )

        return row_to_auth_context(row)

    def set_user_status(
        self,
        user_id: str,
        status_value: str,
    ) -> None:
        status_value = status_value.strip().lower()

        if status_value not in {"active", "disabled"}:
            raise ValueError(
                "User status must be 'active' or 'disabled'."
            )

        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE users
                SET status = ?,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (
                    status_value,
                    utc_now_iso(),
                    user_id,
                ),
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"User not found: {user_id}"
                )


# Global store. It opens SQLite connections per operation, rather than holding
# one connection across FastAPI threads.
auth_store = AuthStore(AUTH_DB_PATH)


# ============================================================================
# Row -> AuthContext
# ============================================================================

def row_to_auth_context(row: sqlite3.Row) -> AuthContext:
    organization_status = row["organization_status"]

    if organization_status != "active":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Organization is not active.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if row["status"] != "active":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is not active.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthContext(
        user_id=str(row["user_id"]),
        name=str(row["name"]),
        email=str(row["email"]),
        organization_id=str(row["organization_id"]),
        organization_name=str(row["organization_name"]),
        role=normalise_role(str(row["role"])),
        status=str(row["status"]),
    )


# ============================================================================
# JWT helpers
# ============================================================================

def create_access_token(
    user_id: str,
    *,
    expires_minutes: int | None = None,
) -> str:
    """
    Create a signed JWT access token.

    `sub` is the authoritative user identity.
    Organization and role are deliberately resolved from the DB during
    authentication so current account state can be enforced.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")

    now = utc_now()

    lifetime_minutes = (
        expires_minutes
        if expires_minutes is not None
        else JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )

    if lifetime_minutes <= 0:
        raise ValueError("expires_minutes must be > 0")

    payload = {
        "sub": user_id,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=lifetime_minutes),
        "jti": str(uuid.uuid4()),
        "type": "access",
    }

    return jwt.encode(
        payload,
        _jwt_secret(),
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Validate a JWT access token and return its claims.

    Validation includes:
        - signature
        - algorithm
        - issuer
        - audience
        - expiration
        - token type
        - required `sub`
    """
    if not token or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={
                "require": ["sub", "exp", "iat", "iss", "aud"],
            },
        )

    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    subject = payload.get("sub")

    if not isinstance(subject, str) or not subject.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has no valid user identity.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


# ============================================================================
# Public authentication helpers
# ============================================================================

def create_user(
    *,
    name: str,
    email: str,
    password: str,
    organization: str,
    role: str,
) -> dict[str, Any]:
    """
    Main registration helper for the future POST /create_user endpoint.

    Creates:
        organization (if it does not exist)
        user
        password hash
        access token

    Returns:
        user_id
        access_token
        token_type
        user/auth context
    """
    if not password:
        raise ValueError("password must be a non-empty string")

    if len(password) < 8:
        raise ValueError(
            "password must contain at least 8 characters"
        )

    password_hash = hash_password(password)

    user = auth_store.create_user(
        name=name,
        email=email,
        password_hash=password_hash,
        organization_name=organization,
        role=role,
    )

    access_token = create_access_token(
        user.user_id
    )

    return {
        "user_id": user.user_id,
        "access_token": access_token,
        "token_type": "bearer",
        "user": user.as_dict(),
    }


def authenticate_user(
    *,
    email: str,
    password: str,
) -> dict[str, Any]:
    """
    Authenticate email/password credentials and issue a fresh JWT.
    """
    user_row = auth_store.get_user_by_email(email)

    # Do not reveal whether the email exists.
    if user_row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(
        password,
        str(user_row["password_hash"]),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = row_to_auth_context(user_row)

    access_token = create_access_token(
        user.user_id
    )

    return {
        "user_id": user.user_id,
        "access_token": access_token,
        "token_type": "bearer",
        "user": user.as_dict(),
    }


def authenticate_token(token: str) -> AuthContext:
    """
    Main token -> authenticated-user function.

    Flow:
        JWT
         ↓
        validate signature/claims
         ↓
        extract user_id from `sub`
         ↓
        load current user state
         ↓
        return AuthContext
    """
    payload = decode_access_token(token)

    user_id = str(payload["sub"]).strip()

    user_row = auth_store.get_user_by_id(user_id)

    if user_row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return row_to_auth_context(user_row)


# ============================================================================
# FastAPI dependency
# ============================================================================

def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> AuthContext:
    """
    FastAPI dependency for protected endpoints.

    Later, API routes can do:

        @app.get("/me")
        def me(
            current_user: AuthContext = Depends(get_current_user),
        ):
            return current_user.as_dict()

    The route does NOT accept user_id/organization/role from the client.
    They come from the validated authentication context.
    """
    return authenticate_token(token)


# ============================================================================
# Small helper for role validation
# ============================================================================

def is_valid_role(role: str) -> bool:
    """Return True when role is one of the four fixed V1 roles."""
    try:
        normalise_role(role)
        return True
    except ValueError:
        return False





# ============================================================================
# Development / smoke-test entry point
# ============================================================================

def main() -> None:
    """
    Minimal local smoke test.

    This does not create a permanent sample user automatically.
    It only confirms the auth database can be initialized.

    Run:
        python authorization.py
    """
    print("Authentication module initialized.")
    print(f"Auth DB: {AUTH_DB_PATH}")
    print(
        "Allowed roles:",
        ", ".join(sorted(ALLOWED_ROLES)),
    )


if __name__ == "__main__":
    main()
