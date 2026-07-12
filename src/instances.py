# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

import logging
import os
import sys

from flasgger import Swagger
from flask import Blueprint, Flask, request, g, current_app
from flask_jwt_extended import JWTManager, verify_jwt_in_request, get_jwt_identity
from supabase import create_client
from datetime import datetime
from zoneinfo import ZoneInfo
from werkzeug.exceptions import HTTPException

from config import Config

# ----------------------------------------------------------------------------------------------- #
# Inicializar instâncias
# ----------------------------------------------------------------------------------------------- #

bp = Blueprint('main', __name__)
jwt = JWTManager()

swagger = Swagger(
    template = {
        "securityDefinitions": {
            "BearerAuth": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "Digite: Bearer <seu_token>"
            }
        }
    })

# ----------------------------------------------------------------------------------------------- #
# Conectar banco de dados
# ----------------------------------------------------------------------------------------------- #

supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

# ----------------------------------------------------------------------------------------------- #
# Configurar logs
# ----------------------------------------------------------------------------------------------- #

LOG_DIR = "logs"

# Configurar registros de logs (local e flask) ----------------------------------------------------

def setup_logging(app: Flask) -> None:

    # definir o layout da mensagem: 
    # [Data/Hora] Nível de Severidade em NomeDoModulo: Mensagem
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )

    # limpar handlers existentes
    app.logger.handlers.clear()
    app.logger.setLevel(logging.INFO)

    # sempre logar em stdout (Vercel captura)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    app.logger.addHandler(stream_handler)

    # SOMENTE LOCAL: criar pasta e arquivo
    if os.getenv("FLASK_ENV") == "development":
        tz_sp = ZoneInfo("America/Sao_Paulo")
        timestamp = datetime.now(tz_sp).strftime("%Y-%m-%d_%H-%M-%S")

        os.makedirs(LOG_DIR, exist_ok = True)
        log_file = f"{LOG_DIR}/app_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)

        app.logger.info("Logger inicializado em arquivo (local)")
    else:
        app.logger.info("Logger inicializado (stdout / Render)")

# Configurar registros de logs (supabase) ---------------------------------------------------------

def register_request_logging(app, supabase):
    tz_sp = ZoneInfo("America/Sao_Paulo")

    IGNORED_PATHS = {
        "/flasgger_static/swagger-ui-standalone-preset.js",
        "/apispec_1.json",
        "/flasgger_static/swagger-ui.css",
        "/flasgger_static/swagger-ui-bundle.js",
        "/static/github.png",
        "/flasgger_static/lib/jquery.min.js",
        "/static/styles.css",
        "/static/question_mark.png",
        '/api/v1/health',
        '/',
        '/flasgger_static/favicon-32x32.png',
        '/static/favicon.png',
        '/favicon.ico',
        '/apidocs'
    }

    # Carregar usuário (ANTES do logging e do after_request)
    @app.before_request
    def load_user():
        try:
            verify_jwt_in_request(optional=True)
            g.user_id = get_jwt_identity()
        except Exception:
            g.user_id = None

    # Log simples em stdout
    @app.before_request
    def log_request():
        g.request_start_time = datetime.now(tz_sp)
        current_app.logger.info(
            f"{request.method} {request.path}"
        )

    # Log estruturado no Supabase
    def log_request_to_supabase_factory(supabase):
        def log_request_to_supabase(response):
            if request.path in IGNORED_PATHS:
                return response

            try:
                supabase.table("api_request_logs").insert({
                    "user_id": g.user_id,
                    "method": request.method,
                    "path": request.path,
                    "status_code": response.status_code,
                }).execute()

            except Exception as e:
                current_app.logger.error(
                    f"Erro ao salvar log no Supabase: {e}"
                )

            return response

        return log_request_to_supabase

    app.after_request(log_request_to_supabase_factory(supabase))

    # Tratamento de exceções
    @app.errorhandler(Exception)
    def handle_exception(e):
        if isinstance(e, HTTPException):
            return e
        current_app.logger.exception(e)
        return {"error": "internal server error"}, 500
