"""Initial registry and durable job queue."""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("projects", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("name", sa.String(120), nullable=False, unique=True), sa.Column("description", sa.Text(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("datasets", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False), sa.Column("name", sa.String(120), nullable=False), sa.Column("root", sa.Text(), nullable=False), sa.Column("digest", sa.String(64), nullable=False), sa.Column("manifest", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("registered_models", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False), sa.Column("name", sa.String(120), nullable=False), sa.Column("path", sa.Text(), nullable=False), sa.Column("digest", sa.String(64), nullable=False), sa.Column("provenance", sa.Text(), nullable=False), sa.Column("classes", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("jobs", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False), sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id"), nullable=False), sa.Column("model_id", sa.Integer(), sa.ForeignKey("registered_models.id"), nullable=False), sa.Column("status", sa.String(20), nullable=False), sa.Column("progress", sa.Integer(), nullable=False), sa.Column("attempt", sa.Integer(), nullable=False), sa.Column("spec", sa.JSON(), nullable=False), sa.Column("result", sa.JSON(), nullable=True), sa.Column("error", sa.Text(), nullable=True), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("started_at", sa.DateTime(timezone=True), nullable=True), sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_jobs_status", "jobs", ["status"])


def downgrade():
    op.drop_index("ix_jobs_status", "jobs")
    op.drop_table("jobs")
    op.drop_table("registered_models")
    op.drop_table("datasets")
    op.drop_table("projects")
