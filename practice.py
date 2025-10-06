print("hello".__doc__.__sizeof__())

print([m for m in dir(str) if m.startswith("__") and m.endswith("__")])
