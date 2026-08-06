#!/usr/bin/env bash
set -e
echo "---APT SOURCES---"
apt-get --version 2>&1 | head -1
echo "---INSTALL OPENJDK---"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>&1 | tail -3
apt-get install -y -qq openjdk-11-jre-headless 2>&1 | tail -5 || apt-get install -y -qq default-jre-headless 2>&1 | tail -5
echo "---JAVA VERIFY---"
java -version 2>&1 | head -3