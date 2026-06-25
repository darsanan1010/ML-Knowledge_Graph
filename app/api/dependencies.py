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

def validate_token(token: str, expected_scope: str):
    ml_secret = os.getenv("CARE_MP_ML_SERVICE_SECRET")
    ml_issuer = os.getenv("CARE_MP_ML_SERVICE_ISSUE", "caremp-api-service")
    ml_audience = os.getenv("CARE_MP_ML_SERVICE_AUD", "caremp-ml-engine-service")
    expected_role = os.getenv("CARE_MP_ML_SERVICE_ROLE", "caremp-api-service")
    
    if not ml_secret:
        raise HTTPException(status_code=500, detail="CARE_MP_ML_SERVICE_SECRET is not configured on the server")

    try:
        payload = jwt.decode(
            token,
            ml_secret,
            algorithms=["HS256"],
            issuer=ml_issuer,
            audience=ml_audience
        )
        if payload.get("role") != expected_role:
            raise HTTPException(status_code=403, detail="Invalid role")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_sync_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return validate_token(credentials.credentials, "resident:sync")
