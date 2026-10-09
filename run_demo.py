import threading
import time
import webbrowser
import uvicorn


def main():
    print("=" * 60)
    print("   TACTICAL REINFORCEMENT LEARNING OPERATOR CONSOLE")
    print("   Modeled after Industrial Multi-Camera Tracking Architecture")
    print("=" * 60)
    print("[*] Launching FastAPI Live Engine at http://127.0.0.1:8000 ...")

    def open_browser():
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(
        "src.api.server:app", host="127.0.0.1", port=8000, reload=False
    )


if __name__ == "__main__":
    main()
