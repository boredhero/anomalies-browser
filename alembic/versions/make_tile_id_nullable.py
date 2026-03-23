"""make tile_id nullable on detections

Revision ID: b001
Revises: e18ee75b19f6
"""
from alembic import op
import sqlalchemy as sa

revision = 'b001'
down_revision = 'e18ee75b19f6'


def upgrade():
    op.alter_column('detections', 'tile_id', existing_type=sa.UUID(), nullable=True)


def downgrade():
    op.alter_column('detections', 'tile_id', existing_type=sa.UUID(), nullable=False)
