import threading
from server import server
from app_routes import app as flask_app

def run_mesa_server():
    server.launch()

def run_flask_app():
    flask_app.run(port=5000)

if __name__ == '__main__':
    mesa_server_thread = threading.Thread(target=run_mesa_server)
    flask_app_thread = threading.Thread(target=run_flask_app)

    mesa_server_thread.start()
    flask_app_thread.start()
