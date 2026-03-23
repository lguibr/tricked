from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from tricked_web.routes import register_routes
from tricked_web.sockets import register_sockets

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

register_routes(app)
background_telemetry_thread = register_sockets(socketio)
