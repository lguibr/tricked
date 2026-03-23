import uvicorn

from tricked_web.app import app

if __name__ == "__main__":
    from tricked_web.state import reset_game
    reset_game()
    uvicorn.run(app, host="0.0.0.0", port=8080)
