#!/bin/bash
gunicorn -w 4 -b 0.0.0.0:$PORT -k uvicorn.workers.UvicornWorker main:app

