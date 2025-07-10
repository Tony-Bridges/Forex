import os
import logging

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_bootstrap import Bootstrap
from flask_login import LoginManager

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('app')

class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy with a custom model class
db = SQLAlchemy(model_class=Base)

# Create the Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "supersecretkey")

# Configure ProxyFix for handling proxy headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1) 

# Configure the database connection
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Configure Bootstrap
bootstrap = Bootstrap(app)

# Configure Login Manager
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Initialize the database
db.init_app(app)

# Import models after database initialization to avoid circular imports
with app.app_context():
    logger.info("Creating database tables...")
    import models
    db.create_all()
    logger.info("Database tables created successfully.")

# Import views after everything is initialized
from views import init_views
init_views(app)