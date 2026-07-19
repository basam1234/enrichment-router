from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_database_url() -> str:
    """Return the database URL from ``DATABASE_URL`` env var, falling back to
    a local SQLite file.  The default is a deliberate seam so production
    can switch to PostgreSQL by changing one string.
    """
    return os.getenv("DATABASE_URL", "sqlite:///./enrichment.db")


class RecordORM(Base):
    """Top-level enrichment record: one company/entity submitted for enrichment.

    Each record stores the name and optional domain, plus the raw request
    payload as JSON so the original input can always be reconstructed.
    """

    __tablename__ = "records"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(Text, nullable=False)
    domain: str | None = Column(Text, nullable=True)
    request_json: str = Column(Text, nullable=False)
    created_at: datetime = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    runs: Mapped[list["EnrichmentRunORM"]] = relationship(
        "EnrichmentRunORM",
        back_populates="record",
        cascade="all, delete-orphan",
        order_by="EnrichmentRunORM.id.desc()",
    )


class EnrichmentRunORM(Base):
    """One execution of the enrichment pipeline against a single record.

    Captures the final status, aggregated cost/latency, and a JSON-serialised
    snapshot of every resolved field so the UI can render results without
    recomputing them.
    """

    __tablename__ = "enrichment_runs"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    record_id: int = Column(Integer, ForeignKey("records.id"), nullable=False)
    status: str = Column(Text, nullable=False)
    total_cost_usd: float = Column(Float, default=0.0, nullable=False)
    total_latency_ms: float = Column(Float, default=0.0, nullable=False)
    resolved_fields_json: str = Column(Text, nullable=False)
    created_at: datetime = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    record: Mapped["RecordORM"] = relationship("RecordORM", back_populates="runs")
    trace_events: Mapped[list["TraceEventORM"]] = relationship(
        "TraceEventORM",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="TraceEventORM.id.asc()",
    )


class TraceEventORM(Base):
    """A single node in the enrichment pipeline's execution trace.

    Stored inline with the run so consumers can reconstruct what happened
    step-by-step without external observability tools.
    """

    __tablename__ = "trace_events"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    run_id: int = Column(Integer, ForeignKey("enrichment_runs.id"), nullable=False)
    node: str = Column(Text, nullable=False)
    detail: str = Column(Text, nullable=False)
    created_at: datetime = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    run: Mapped["EnrichmentRunORM"] = relationship(
        "EnrichmentRunORM", back_populates="trace_events"
    )


def init_db(database_url: str | None = None) -> Engine:
    """Create all tables on the given (or default) database URL and return
    the engine.  Idempotent: safe to call multiple times.

    Returning the engine lets tests construct an in-memory SQLite DB and
    hold a reference to it (otherwise ``:memory:`` databases vanish per
    connection).  For tests, callers should pass ``"sqlite:///:memory:"``.
    """
    url = database_url or get_database_url()
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine
