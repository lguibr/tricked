import eventlet

eventlet.monkey_patch()

if __name__ == "__main__":
    from tricked_web.app import app, background_telemetry_thread, socketio
    from tricked_web.state import reset_game

    reset_game()
    socketio.start_background_task(background_telemetry_thread)
    socketio.run(app, debug=True, port=8080, use_reloader=False, host="0.0.0.0")
