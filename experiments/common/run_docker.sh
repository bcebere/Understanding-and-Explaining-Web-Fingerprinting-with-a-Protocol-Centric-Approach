index=$@
if [[ -z "$index" ]];
then
    index="0"
fi

mkdir -p shared_workspace
chmod 777 shared_workspace
# CHROME
echo "Creating interface $index"
sel_port=`expr ${index} + ${index} + 4444`
http_port=`expr ${index} + ${index} + 7900`

## cleanup
docker stop "selenium-chrome${index}"
docker rm -f "selenium-chrome${index}"
docker network prune -f

## Start Chrome
docker network create --opt com.docker.network.bridge.name="veth_chrome${index}" "veth-chrome${index}"
docker run -p ${sel_port}:4444 -p ${http_port}:7900 --shm-size="2g" --dns=127.0.0.1 --name="selenium-chrome${index}" --network "veth-chrome${index}"  -v ${PWD}/shared_workspace:/tmp/shared_workspace -e SSLKEYLOGFILE=/tmp/shared_workspace/ssl_log_key_chrome${index}.log --rm -dit "selenium-chrome"

# FIREFOX
sel_port=`expr ${index} + ${index} + 4445`
http_port=`expr ${index} + ${index} + 7901`

docker stop "selenium-firefox${index}"
docker rm -f "selenium-firefox${index}"
docker network prune -f

# Start Firefox
docker network create --opt com.docker.network.bridge.name="veth_firefox${index}" "veth-firefox${index}"
docker run -p ${sel_port}:4444 -p ${http_port}:7901 --shm-size="2g" --dns=127.0.0.1 --name="selenium-firefox${index}" --network "veth-firefox${index}"  -v ${PWD}/shared_workspace:/tmp/shared_workspace -e SSLKEYLOGFILE=/tmp/shared_workspace/ssl_log_key_ff${index}.log --rm -dit "selenium-firefox"
