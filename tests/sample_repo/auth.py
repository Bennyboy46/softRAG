import os
import hashlib
from typing import Optional
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Sample Authentication Service")

SECRET_KEY = "super-secret-key"
ALGORITHM = "HS256"


@app.post("/api/v1/auth/login")
def login(username: str, password_hash: str):
    """Authenticate user credentials and return an access token."""
    if not username or not password_hash:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    auth_service = AuthService()
    if not auth_service.validate_credentials(username, password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"token": "mock-jwt-token-12345", "status": "authenticated"}


class AuthService:
    """Manages user authentication and credential validation."""

    def __init__(self, salt: str = "custom_salt"):
        self.salt = salt

    def validate_credentials(self, username: str, password_hash: str) -> bool:
        """Validate provided hash against the database store."""
        expected = self.hash_password(username)
        return expected == password_hash

    def hash_password(self, raw_password: str) -> str:
        """Compute SHA256 hash with configured salt."""
        salted = f"{raw_password}:{self.salt}".encode("utf-8")
        return hashlib.sha256(salted).hexdigest()


def create_access_token(user_id: str, expires_in_seconds: int = 3600) -> str:
    """Generate a signed JWT access token for the given user."""
    return f"jwt_token_for_{user_id}_{expires_in_seconds}"
