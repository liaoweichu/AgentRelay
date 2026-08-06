#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive
echo "---INSTALL OPENJDK-17---"
apt-get install -y -qq openjdk-17-jre-headless 2>&1 | tail -5
echo "---JAVA17 VERIFY---"
ls -d /usr/lib/jvm/java-17-openjdk-amd64 2>&1
/usr/lib/jvm/java-17-openjdk-amd64/bin/java -version 2>&1 | head -2