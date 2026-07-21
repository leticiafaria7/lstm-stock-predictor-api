# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from config import Config, BASE_DIR

from src.instances import bp, swagger

import os

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

# ----------------------------------------------------------------------------------------------- #
# Executar o app localmente
# ----------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    with app.app_context():
        print(BASE_DIR)
        print(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # app.run(debug=True)

