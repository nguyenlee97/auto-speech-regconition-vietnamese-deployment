SHELL := /bin/bash

.PHONY: help build up up-caddy down logs curl test

help:
	@echo "Targets: build, up, up-caddy, down, logs, curl, test"

build:
	docker compose build

up:
	cp -n .env.example .env || true
	docker compose up -d

up-caddy:
	cp -n .env.example .env || true
	docker compose -f docker-compose.yml -f docker-compose.caddy.yml up -d

down:
	docker compose down

logs:
	docker compose logs -f vi-asr

curl:
	@echo "curl -X POST \"http://localhost:8080/v1/transcribe?decoding_method=modified_beam_search&num_active_paths=15\" \\" \
	      "-H \"Authorization: Bearer $${API_KEY:-changeme}\" -F \"file=@test_wavs/vietnamese/0.wav\""

test:
	curl -sSf http://localhost:8080/readyz && echo OK || (echo FAIL && exit 1)