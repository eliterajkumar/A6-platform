from flask import Blueprint
from src.controllers.model_controller import fine_tune, generate

model_bp = Blueprint('model_bp', __name__)

model_bp.route('/fine-tune', methods=['POST'])(fine_tune)
model_bp.route('/generate', methods=['POST'])(generate)
