# GEANT4 build + run environment for the positron transport driver.
#
# Builds GEANT4 from source (self-contained, reproducible) and compiles
# run_geant4.cc against it. input/ and output/geant4/ are bind-mounted
# at run time (see run_geant4.sh) -- nothing project-specific is baked
# into the image.

FROM ubuntu:22.04 AS geant4-build

ARG GEANT4_VERSION=11.3.2
ARG JOBS
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates \
        libexpat1-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/build
RUN wget -q "https://geant4-data.web.cern.ch/releases/geant4-v${GEANT4_VERSION}.tar.gz" \
    && tar xzf "geant4-v${GEANT4_VERSION}.tar.gz" \
    && rm "geant4-v${GEANT4_VERSION}.tar.gz"

RUN mkdir -p /opt/build/build && cd /opt/build/build \
    && cmake ../geant4-v${GEANT4_VERSION} \
        -DCMAKE_INSTALL_PREFIX=/opt/geant4 \
        -DGEANT4_INSTALL_DATA=ON \
        -DGEANT4_BUILD_MULTITHREADED=OFF \
        -DGEANT4_USE_GDML=OFF \
        -DGEANT4_USE_QT=OFF \
        -DGEANT4_USE_OPENGL_X11=OFF \
        -DGEANT4_USE_SYSTEM_EXPAT=ON \
        -DGEANT4_USE_SYSTEM_ZLIB=ON \
    && make -j"${JOBS:-$(nproc)}" \
    && make install \
    && rm -rf /opt/build

FROM ubuntu:22.04

ARG GEANT4_VERSION=11.3.2
ARG JOBS
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libexpat1-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=geant4-build /opt/geant4 /opt/geant4

WORKDIR /src
COPY run_geant4.cc CMakeLists.txt /src/
RUN cmake -B /src/build -S /src \
        -DGeant4_DIR=/opt/geant4/lib/Geant4-${GEANT4_VERSION} \
    && cmake --build /src/build -j"${JOBS:-$(nproc)}" \
    && cp /src/build/run_geant4 /usr/local/bin/ \
    && rm -rf /src/build

WORKDIR /work
ENTRYPOINT ["/bin/bash", "-c", "source /opt/geant4/bin/geant4.sh && exec run_geant4"]
