
build:
	docker build -t vrd-rakede-dev .

run: build
	docker run -u $(id -u):$(id -g) --runtime=nvidia -it --rm -v $(PWD):/src/:ro \
		-v /mnt/renumics-research:/mnt/renumics-research:ro \
		-v $(PWD)/weights:/src/weights/:rw \
		-v $(PWD)/snapshots:/src/snapshots/:rw \
		-v $(PWD)/logs:/src/logs/:rw \
		vrd-rakede-dev /bin/bash
