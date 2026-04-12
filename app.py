import os
import sys
import uvicorn

# Ensure the local environment package is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), 'metamavericks_env'))

from server.app import app

if __name__ == "__main__":
    # Hugging Face Spaces uses port 7860 by default
    # Local development often uses 8000
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting MetaMavericks OpenEnv Server on port {port}...")
    print(f"Dashboard available at: http://localhost:{port}/web")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
