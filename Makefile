
build:
	docker build -t vrd-rakede-dev .

run: build
	docker run --runtime=nvidia -it --rm -v $(PWD):/src/:ro \
		-v /mnt/renumics-research:/mnt/renumics-research:ro \
		vrd-rakede-dev /bin/bash
