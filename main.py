# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from config import Config, BASE_DIR

from src.instances import bp, swagger

from src.api import api_endpoints, home_numbers_and_plots

import os

from src.monitoring_middleware import register_monitoring

# ----------------------------------------------------------------------------------------------- #
# Inicializações
# ----------------------------------------------------------------------------------------------- #

# iniciar aplicação
app = Flask(__name__,
            template_folder = os.path.join(BASE_DIR, "src", "templates"),
            static_folder = os.path.join(BASE_DIR, "src", "static")
            )

# instanciar configs
app.config.from_object(Config)

# iinicializar a documentação com swagger
swagger.init_app(app)

# registrar as rotas
app.register_blueprint(bp)

# monitoramento prometheus
register_monitoring(app)

# ----------------------------------------------------------------------------------------------- #
# Executar o app localmente
# ----------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)

