
for index in 1 2 3 4;
do
    echo $index
    docker stop "selenium-chrome${index}"
    docker stop "selenium-firefox${index}"
    docker rm -f "selenium-chrome${index}"
    docker rm -f "selenium-firefox${index}"
    docker network prune -f
done
