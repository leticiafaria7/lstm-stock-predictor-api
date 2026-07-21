# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

from flasgger import Swagger
from flask import Blueprint
from flask_jwt_extended import JWTManager
from supabase import create_client

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
