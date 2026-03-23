# Tricked Web

Python-based API connecting the heavy computational node (`tricked`) to the front-facing dashboards (`tricked UI/Svelte`).

## Components
- `server.py` & `app.py`: Flask and SocketIO definitions handling bridging.
- `sockets.py`: Implements robust event-driven channels mirroring the live buffer of the game loop.
- `state.py`: Encapsulates threading-safe tracking metrics.
