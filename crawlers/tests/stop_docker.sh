index=$@
if [[ -z "$index" ]];
then
    index="0"
fi

docker stop "selenium-chrome${index}"
docker rm -f "selenium-chrome${index}"

docker stop "selenium-firefox${index}"
docker rm -f "selenium-firefox${index}"
docker network prune -f
