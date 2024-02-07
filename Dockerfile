FROM node:21-bookworm

RUN sed -ri 's#URIs: http://deb.debian.org#URIs: http://mirrors.tuna.tsinghua.edu.cn#g' /etc/apt/sources.list.d/debian.sources && \
	apt-get update && apt-get install -y apt-transport-https ca-certificates && sed -ri 's#URIs: http://#URIs: https://#g' /etc/apt/sources.list.d/debian.sources && \
	apt-get update && apt upgrade -y

RUN apt-get update -y && apt-get install -y python3-venv && \
	git clone https://github.com/ccoder64/search_rag.git --depth=1 --branch=main /app && \
	cd /app && \
	python3 -m venv venv && bash -c "source venv/bin/activate && pip3 install -r requirements.txt" && \
	cd web && npm i && npm run build && rm -rf node_modules .next && npm cache clean --force


WORKDIR /app

EXPOSE 8800

ENTRYPOINT ["bash", "-c",  "cd /app; source venv/bin/activate; exec python3 search_rag.py"]
