from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import re
from datetime import datetime, timezone
import hashlib
import secrets
import logging
from db.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

class UserAuthRequest(BaseModel):
    email: str
    password: str

EMAIL_REGEX = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

class AuthResponse(BaseModel):
    success: bool
    message: str
    email: str

def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    # OWASP 2024 guidance: PBKDF2-HMAC-SHA256 with ≥ 600,000 iterations.
    # Preferred for new systems: Argon2id (not yet in stdlib; would require argon2-cffi).
    pw_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        iterations=600_000
    ).hex()
    return pw_hash, salt

def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    computed_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, stored_hash)

@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserAuthRequest):
    db = get_db()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable. Please ensure MongoDB is running."
        )

    email = payload.email.lower().strip()
    if not EMAIL_REGEX.match(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format. Please provide a valid email address."
        )
    if len(payload.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long."
        )

    # Check if user already exists
    existing_user = await db.users.find_one({"email": email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists. Please sign in."
        )

    # Hash password with secure per-user salt
    pw_hash, salt = hash_password(payload.password)

    user_doc = {
        "email": email,
        "password_hash": pw_hash,
        "salt": salt,
        "created_at": datetime.now(timezone.utc),
        "last_login": datetime.now(timezone.utc)
    }

    try:
        await db.users.insert_one(user_doc)
        logger.info(f"New user registered successfully: {email}")
        return AuthResponse(
            success=True,
            message="Account created successfully.",
            email=email
        )
    except Exception as e:
        logger.error(f"Failed to register user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during account creation."
        )

@router.post("/login", response_model=AuthResponse)
async def login(payload: UserAuthRequest):
    db = get_db()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable. Please ensure MongoDB is running."
        )

    email = payload.email.lower().strip()
    user = await db.users.find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    if not verify_password(payload.password, user["password_hash"], user["salt"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # Update last login timestamp
    await db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.now(timezone.utc)}}
    )

    logger.info(f"User authenticated successfully: {email}")
    return AuthResponse(
        success=True,
        message="Authentication successful.",
        email=email
    )
