from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.websocket_handler import handle_audio_stream

app = FastAPI(title="Voice Edge KWS - ASR Server")

@app.get("/")
def read_root():
    return {"status": "ASR Server is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await handle_audio_stream(websocket)
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
