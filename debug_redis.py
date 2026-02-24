import redis, json

r = redis.from_url("redis://redis:6379/0", decode_responses=True)
entries = r.xrange("mcn:runs", count=2)
for _id, fields in entries:
    print("=== entry", _id)
    for k, v in fields.items():
        print(f"  {k!r}: {v!r}  (type={type(v).__name__})")
    print()
    # Try json.loads on each value
    print("  After json.loads:")
    for k, v in fields.items():
        try:
            parsed = json.loads(v)
            print(f"    {k}: {parsed!r}  (parsed type={type(parsed).__name__})")
        except Exception:
            print(f"    {k}: (not valid JSON) {v!r}")
    break
