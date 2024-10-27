def not_allowed(func): 
    def wrapper(*args, **kwargs): 
        raise PermissionError(f"The method {func.__name__} is not allowed in this subclass.")
    return wrapper