FROM ubuntu:22.04

# Install Java
RUN apt-get update && apt-get install -y openjdk-11-jdk


# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip


# Install curl
RUN apt-get update && apt-get install -y curl


# Downloading and installing Maven
# 1- Define a constant with the version of maven you want to install
ARG MAVEN_VERSION=3.9.8         

# 2- Define a constant with the working directory
ARG USER_HOME_DIR="/root"

# 4- Define the URL where maven can be downloaded from
ARG BASE_URL=https://apache.osuosl.org/maven/maven-3/${MAVEN_VERSION}/binaries

# 5- Create the directories, download maven, validate the download, install it, remove downloaded file and set links
RUN mkdir -p /usr/share/maven /usr/share/maven/ref \
    && echo "Downlaoding maven" \
    && curl -fsSL -o /tmp/apache-maven.tar.gz ${BASE_URL}/apache-maven-${MAVEN_VERSION}-bin.tar.gz \
    \
    && echo "Unziping maven" \
    && tar -xzf /tmp/apache-maven.tar.gz -C /usr/share/maven --strip-components=1 \
    \
    && echo "Cleaning and setting links" \
    && rm -f /tmp/apache-maven.tar.gz \
    && ln -s /usr/share/maven/bin/mvn /usr/bin/mvn

# 6- Define environmental variables required by Maven, like Maven_Home directory and where the maven repo is located
ENV MAVEN_HOME /usr/share/maven
ENV MAVEN_CONFIG "$USER_HOME_DIR/.m2"

# Set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python



RUN pip install numpy
RUN pip install "scyjava"
RUN pip install "requests"
RUN pip install "xarray-multiscale"
RUN pip install "aicsimageio[nd2, dv, base-imageio]==4.14"
RUN pip install "bioformats_jar"
RUN pip install "readlif>=0.6.4"
RUN pip install "aicspylibczi>=3.1.1"


COPY test.tiff /tmp
COPY z.py /tmp
WORKDIR /tmp
RUN python z.py

RUN pip install "arkitekt-next[all]==0.8.45"



# Install Arbeid
RUN mkdir /workspace
ADD . /workspace
WORKDIR /workspace


