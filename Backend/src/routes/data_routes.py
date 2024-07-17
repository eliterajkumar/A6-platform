from flask import Blueprint
from src.controllers.data_controller import upload_data

data_bp = Blueprint('data_bp', __name__)

data_bp.route('/upload', methods=['POST'])(upload_data)
