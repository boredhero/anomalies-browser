"""Validation workflow endpoints: confirm/reject/annotate detections."""

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from magic_eyes.db.engine import get_session
from magic_eyes.db.models import ValidationEvent, ValidationVerdict

router = APIRouter(tags=["validation"])


class ValidationRequest(BaseModel):
    verdict: ValidationVerdict
    notes: str | None = None


@router.post("/detections/{detection_id}/validate")
async def validate_detection(
    detection_id: uuid.UUID,
    body: ValidationRequest,
    session: AsyncSession = Depends(get_session),
):
    event = ValidationEvent(
        detection_id=detection_id,
        verdict=body.verdict,
        notes=body.notes,
    )
    session.add(event)
    await session.commit()
    return {"status": "ok", "verdict": body.verdict.value}
