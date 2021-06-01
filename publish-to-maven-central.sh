#!/bin/bash

set -x

OSSRH_USER=$1
OSSRH_PASS=$2

echo 'Starting Maven build and publish to maven central'
mvn -V -B clean deploy -DskipTests -P "gpg-sign-deploy" -s settings.xml -Dossrh_user=$OSSRH_USER -Dossrh_pass=$OSSRH_PASS
echo 'Finished Maven build and publish to maven central'
