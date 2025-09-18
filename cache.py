cache = {}

def get_cached_response(prompt):
    return cache.get(prompt)

def set_cached_response(prompt, response):
    cache[prompt] = response
