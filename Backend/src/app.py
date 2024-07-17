from flask import Flask
from src.routes.data_routes import data_bp
from src.routes.model_routes import model_bp
from src.config.database import db

app = Flask(__name__)
app.register_blueprint(data_bp, url_prefix='/data')
app.register_blueprint(model_bp, url_prefix='/model')

if __name__ == "__main__":
    app.run(debug=True)
