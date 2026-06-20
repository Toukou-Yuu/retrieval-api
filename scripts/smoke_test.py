import httpx


def main() -> None:
    response = httpx.get("http://127.0.0.1:8300/v1/health", timeout=5.0)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
