import os
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
workspace_root = Path(__file__).parent.parent.parent
env_path = workspace_root / "env1.env"
load_dotenv(env_path)

security = HTTPBearer()


def validate_token(token: str, expected_scope: str | None = None):
    """
    Validates a CareMP service-to-service JWT.

    Checks:
      - Signature (CARE_MP_ML_SERVICE_SECRET)
      - Issuer, Audience, Expiry (via PyJWT)
      - Role claim matches CARE_MP_ML_SERVICE_ROLE
      - Scope claim matches expected_scope (if provided)
    """
    ml_secret = os.getenv("CARE_MP_ML_SERVICE_SECRET")
    ml_issuer  = os.getenv("CARE_MP_ML_SERVICE_ISSUE", "caremp-api-service")
    ml_audience = os.getenv("CARE_MP_ML_SERVICE_AUD",  "caremp-ml-engine-service")
    expected_role = os.getenv("CARE_MP_ML_SERVICE_ROLE", "caremp-api-service")

    if not ml_secret:
        raise HTTPException(
            status_code=500,
            detail="CARE_MP_ML_SERVICE_SECRET is not configured on the server",
        )

    try:
        payload = jwt.decode(
            token,
            ml_secret,
            algorithms=["HS256"],
            issuer=ml_issuer,
            audience=ml_audience,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Fix #6 — role check (was always present)
    if payload.get("role") != expected_role:
        raise HTTPException(status_code=403, detail="Invalid role")

    # Fix #6 — scope check now actually enforced (was silently ignored before)
    if expected_scope is not None:
        token_scope = payload.get("scope", "")
        if expected_scope not in token_scope.split():
            raise HTTPException(
                status_code=403,
                detail=f"Token missing required scope: {expected_scope}",
            )

    return payload


def get_sync_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    return validate_token(credentials.credentials, expected_scope="resident:sync")
