#!/bin/bash

set -ex

source .env

mongosh "mongodb://${MONGO_USERNAME}:${MONGO_PASSWORD}@127.0.0.1:27018/fao_impact_monitor?directConnection=true&authSource=admin"

