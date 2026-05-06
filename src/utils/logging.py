"""
Logging estruturado com structlog.

O desafio exige "logging estruturado (sem print())". Structlog produz logs
em formato JSON, que sao faceis de parsear por ferramentas como CloudWatch,
ELK Stack, Datadog, etc.

obs Java:
    structlog é como o SLF4J + Logback com JsonLayout.
    Em vez de logger.info("User {} logged in", userId), faz:
    logger.info("user_logged_in", user_id=userId)
    E a sada é JSON: {"event": "user_logged_in", "user_id": 123, ...}

Uso:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("evento", user_id=123, action="predict")
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configura o logging estruturado para toda a aplicação."""

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Retorna um logger estruturado com o nome do módulo."""
    return structlog.get_logger(name)
