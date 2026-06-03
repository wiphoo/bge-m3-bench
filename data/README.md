# Reference datasets

Pre-generated benchmark input files hosted on Cloudflare R2.

## Restaurant (Thai/English)

```text
Base URL: https://public-assets.wiphoo.dev/datasets/restaurants/v1/
```

| File | Rows | Approx tokens (avg) |
|---|---|---|
| `restaurant_smoke_20.txt` | 20 | 218 |
| `restaurant_t32_1000.txt` | 1000 | 58 |
| `restaurant_t64_1000.txt` | 1000 | 87 |
| `restaurant_t128_1000.txt` | 1000 | 155 |
| `restaurant_t256_1000.txt` | 1000 | 263 |
| `restaurant_t512_1000.txt` | 1000 | 522 |
| `restaurant_mixed_lengths_1000.txt` | 1000 | 217 |

Usage:

```bash
uv run bge-m3-bench --address localhost:50051 \
  --texts https://public-assets.wiphoo.dev/datasets/restaurants/v1/restaurant_t128_1000.txt \
  --batch-size 16 --concurrency 4 --warmup-sec 10 --duration-sec 60
```
