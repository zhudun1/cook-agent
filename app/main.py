# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import (
    conversation,
    auth,
    personal_docs,
    user,
    evaluation,
    llm_stats,
    agent,
    diet,
    memory,
)
from app.config import settings
from app.database.session import init_db, close_db
from app.database.document_repository import DocumentRepository
from app.services.auth_service import auth_service
from app.services.rag_service import rag_service_instance
from app.security.middleware.rate_limiter import rate_limiter
from app.security.sanitizer import setup_secure_logging
from app.security.audit import audit_logger
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Setup secure logging with sensitive data filtering
setup_secure_logging()




def _load_config_prompt_registry() -> dict:
    """从 config.yml 加载 prompt_registry 段。"""
    import yaml

    try:
        with open("config.yml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("prompt_registry", {}) or {}
    except Exception:
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup

    # Security check: Validate JWT secret key
    if not settings.JWT_SECRET_KEY:
        logger.error("SECURITY ERROR: JWT_SECRET_KEY environment variable is not set!")
        logger.error(
            "Please set JWT_SECRET_KEY in your .env file or environment variables."
        )
        raise RuntimeError("JWT_SECRET_KEY must be configured for security")

    # 结构化 JSON 日志（traceId 全链路）
    try:
        from app.telemetry.logger import configure_from_settings

        configure_from_settings()
    except Exception as e:
        logger.warning("Failed to setup structured logging: %s", e)

    # P3 Prompt 版本注册（从 config.yml prompt_registry 段加载）
    try:
        from app.prompts.registry import prompt_registry

        pr_data = _load_config_prompt_registry()
        for name, versions in pr_data.items():
            for v in versions:
                prompt_registry.register(
                    name=name,
                    content=v.get("content", ""),
                    version=v.get("version", "v1"),
                    weight=float(v.get("weight", 1.0)),
                    description=v.get("description", ""),
                )
                logger.info(
                    "Registered prompt %s/%s", name, v.get("version", "v1")
                )
    except Exception as e:
        logger.warning("Prompt registry setup skipped: %s", e)

    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized.")

    # Initialize Agent module (registers default agent, tools, skills)
    logger.info("Initializing Agent module...")
    from app.agent import setup_agent_module, setup_mcp_servers

    setup_agent_module()
    logger.info("Agent module initialized.")

    # Register MCP servers (async)
    logger.info("Registering MCP servers...")
    try:
        await setup_mcp_servers()
        logger.info("MCP servers registered.")
    except Exception as e:
        logger.warning(f"Failed to register MCP servers: {e}")

    # Initialize metadata cache (load all global + user metadata at startup)
    logger.info("Initializing metadata cache...")
    await DocumentRepository.init_all_metadata_cache()
    logger.info("Metadata cache initialized.")

    # Clear Redis cache on startup and initialize rate limiter
    if (
        rag_service_instance.cache_manager
        and rag_service_instance.cache_manager.redis_client
    ):
        # Initialize rate limiter with Redis client
        rate_limiter.set_redis(rag_service_instance.cache_manager.redis_client)
        logger.info("Rate limiter initialized with Redis.")
        # Initialize auth service with Redis client for login tracking
        auth_service.set_redis(rag_service_instance.cache_manager.redis_client)
        logger.info("Auth service initialized with Redis for login tracking.")

    yield
    # Shutdown
    logger.info("Closing database connections...")
    await close_db()
    logger.info("Database connections closed.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="The backend API for the CookHero intelligent dietary assistant.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    """
    traceId 全链路追踪中间件。

    读取上游 X-Trace-Id 请求头（无则自动生成），写入 contextvars，
    使同一请求内所有 LLM 调用/工具调用/结构化日志共享同一个 traceId；
    响应头回传 X-Trace-Id 便于前端/网关串联。
    """
    from app.telemetry.trace import set_trace_id, clear_trace_id

    header_name = settings.telemetry.trace.header_name
    trace_id = request.headers.get(header_name)
    if not trace_id and settings.telemetry.trace.auto_generate:
        import uuid

        trace_id = uuid.uuid4().hex
    if trace_id:
        set_trace_id(trace_id)

    response = await call_next(request)
    response.headers[header_name] = trace_id or ""
    clear_trace_id()
    return response


EXEMPT_PATHS = {
    f"{settings.API_V1_STR}/auth/login",
    f"{settings.API_V1_STR}/auth/register",
}


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # Add rate limit headers if available
    if hasattr(request.state, "rate_limit_remaining"):
        response.headers["X-RateLimit-Remaining"] = str(
            request.state.rate_limit_remaining
        )
        response.headers["X-RateLimit-Limit"] = str(request.state.rate_limit_limit)

    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware using Redis."""
    # Check rate limit
    rate_limit_response = await rate_limiter.check_rate_limit(request)
    if rate_limit_response:
        # Log rate limit exceeded
        audit_logger.rate_limit_exceeded(
            request=request,
            user_id=getattr(request.state, "user_id", None),
            endpoint=str(request.url.path),
        )
        return rate_limit_response

    return await call_next(request)


@app.middleware("http")
async def auth_gateway(request: Request, call_next):
    """Simple gateway: require JWT for all routes except login/register/docs/root."""
    path = request.url.path
    if (
        path in EXEMPT_PATHS
        or path == "/"
        or path.startswith("/docs")
        or path.startswith("/redoc")
        or path.startswith("/openapi")
        or path.startswith("/static")
    ):
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return JSONResponse(status_code=401, content={"detail": "需要登录"})

    token = auth_header.split(" ", 1)[1].strip()
    identity = auth_service.decode_token(token)
    if not identity or not identity.get("username"):
        return JSONResponse(
            status_code=401, content={"detail": "登录已失效，请重新登录"}
        )

    # Attach user info to request state for downstream use (e.g., filtering by user)
    request.state.username = identity.get("username")
    request.state.user_id = identity.get("user_id")

    print(request.state.user_id)
    print(request.state.username)

    return await call_next(request)


# Include the API routers
app.include_router(
    conversation.router, prefix=settings.API_V1_STR, tags=["Conversation"]
)
app.include_router(auth.router, prefix=settings.API_V1_STR, tags=["Auth"])
app.include_router(user.router, prefix=settings.API_V1_STR, tags=["User"])
app.include_router(
    personal_docs.router, prefix=settings.API_V1_STR, tags=["KnowledgeBase"]
)
app.include_router(evaluation.router, prefix=settings.API_V1_STR, tags=["Evaluation"])
app.include_router(
    llm_stats.router, prefix=settings.API_V1_STR, tags=["LLM Statistics"]
)
app.include_router(agent.router, prefix=settings.API_V1_STR, tags=["Agent"])
app.include_router(diet.router, prefix=settings.API_V1_STR, tags=["Diet"])
app.include_router(memory.router, prefix=settings.API_V1_STR, tags=["Memory"])


@app.get("/")
async def root():
    """
    Root endpoint to check API status.
    """
    return {"message": "Welcome to CookHero API!"}
