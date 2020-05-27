#!/bin/sh

git status
git add --all
git commit -m "sync jwwangchn codes on $1" -a
git push origin feature/CenterMapOBB