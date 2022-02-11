#!/bin/sh

while read line
do
    printf "%s\n" "$line"
done < "$1"
