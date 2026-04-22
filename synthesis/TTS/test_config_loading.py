from config import config

print(f"ZMQ Bind: {config.zmq_output_bind}")
print(f"Cleaning Blocks Found: {len(config.cleaning_blocks)}")
for rule in config.cleaning_blocks:
    print(f"  - {rule.name}: {rule.start_tag} -> {rule.end_tag}")

print(f"Cleaning Replacements Found: {len(config.cleaning_replacements)}")
for k, v in config.cleaning_replacements.items():
    print(f"  - {repr(k)} -> {repr(v)}")
