#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive
echo "---INSTALL OPENJDK-21---"
apt-get install -y -qq openjdk-21-jre-headless 2>&1 | tail -5
echo "---JAVA21 VERIFY---"
ls -d /usr/lib/jvm/java-21-openjdk-amd64 2>&1
/usr/lib/jvm/java-21-openjdk-amd64/bin/java -version 2>&1 | head -2
echo "---DEFAULT JAVA NOW---"
update-alternatives --set java /usr/lib/jvm/java-21-openjdk-amd64/bin/java 2>&1 | tail -2
java -version 2>&1 | head -2