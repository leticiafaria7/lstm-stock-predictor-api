# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from config import Config, BASE_DIR

from src.instances import bp, swagger, jwt, supabase, setup_logging, register_request_logging

from src.sync_database import sync_database
from src.utils import tickers

import os

from src.api import api_endpoints, home_numbers_and_plots

# ----------------------------------------------------------------------------------------------- #
# Inicializações
# ----------------------------------------------------------------------------------------------- #

# iniciar aplicação
app = Flask(__name__,
            template_folder = os.path.join(BASE_DIR, "src", "templates"),
            static_folder = os.path.join(BASE_DIR, "src", "static")
            )

app.config.from_object(Config)

# inicializar as instâncias no app
swagger.init_app(app)
# jwt.init_app(app)
# setup_logging(app)
register_request_logging(app, supabase)

# registrar as rotas
app.register_blueprint(bp)

# ----------------------------------------------------------------------------------------------- #
# Executar o app localmente
# ----------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    with app.app_context():

        # inicia a API
        app.run(debug=True)