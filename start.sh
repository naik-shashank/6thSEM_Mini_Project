#!/bin/bash
PORT=${PORT:-10000}
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:$PORT --timeout 120 main:app

