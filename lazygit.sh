#!/bin/bash

BLUE='\033[0;34m'
RESET='\033[0m' # No Color

git status
echo $'\n'
echo -e ${BLUE}Type commit message:${RESET}
read message

git add .
git commit -m "$message"

echo $'\n'
echo -e ${BLUE}Do you want to push to master "(y/n):"${RESET}
read answer

if [ $answer = "y" ]
    then
    git push origin master
fi

