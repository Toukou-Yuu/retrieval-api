# retrieval-api

Local retrieval service for agent systems. It accepts documents from upstream knowledge services, builds keyword and vector indexes, and returns structured evidence through HTTP APIs.

## Run

```bash
docker compose up -d
```

Default API endpoint:

```text
http://127.0.0.1:8300
```

API documentation is available in `docs/api.md` and in the service OpenAPI page at `/docs`.
