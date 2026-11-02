# CPU orchestration only. GPU packages and real model weights are not installed.
FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60
RUN apt-get update && apt-get install -y --no-install-recommends git procps ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY . /src
WORKDIR /src
ENV LADDER_ROOT=/results/mock-work
CMD ["bash", "tests/mock/run_linux.sh"]
