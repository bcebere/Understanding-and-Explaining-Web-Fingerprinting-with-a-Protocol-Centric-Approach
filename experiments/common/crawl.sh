#!/bin/bash

index=$@
if [[ -z "$index" ]];
then
    index="0"
fi

bash ./run_docker.sh $index
sleep 5

sudo -E python3 ./crawl.py --offset=$index
