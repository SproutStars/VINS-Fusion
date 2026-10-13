.PHONY: help build clean
help:
	@echo "make build: build the ROS 2 Jazzy image"
	@echo "make clean: remove the image"
build:
	docker build --tag vins-fusion:jazzy -f Dockerfile ..
clean:
	docker image rm vins-fusion:jazzy
