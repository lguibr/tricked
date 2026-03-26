import os
import sys

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
import uvicorn

from tricked_web.app import app

if __name__ == "__main__":
    from tricked_web.state import reset_game

    reset_game()
    uvicorn.run(app, host="0.0.0.0", port=8080)
