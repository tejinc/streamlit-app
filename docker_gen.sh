#!/bin/bash

#build docker image
docker build . -t streamlit-app

#run docker image, 4545 is the docker internal port
# 4546 is local host port
docker run -it --rm -p 4546:4545 streamlit-app