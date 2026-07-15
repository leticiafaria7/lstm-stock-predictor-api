# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------------------------- #
# Classe Config
# ----------------------------------------------------------------------------------------------- #

class Config:

    # configurações de segurança
    SECRET_KEY = os.getenv("SECRET_KEY")

    # caching básico
    CACHE_TYPE = 'simple'

    # título e versão da doc interativa
    SWAGGER = {
        'title': 'API para prever fechamento de ações com modelo LSTM',
        'uiversion': 3
    }
    
    # configurações supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    