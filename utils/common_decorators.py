import time

def count_calls(cls):
    class Wrapper(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.call_counts = {}
            self.total_count = 0

        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if callable(attr) and name not in ['__getattribute__', 'call_counts', 'total_count']:
                if name not in self.call_counts:
                    self.call_counts[name] = 0

                def wrapped_method(*args, **kwargs):
                    self.call_counts[name] += 1
                    self.total_count += 1
                    return attr(*args, **kwargs)

                return wrapped_method

            return attr

    return Wrapper


def measure_execution_time(cls):
    class WrapperClass:
        def __init__(self, *args, **kwargs):
            self.wrapped_instance = cls(*args, **kwargs)
        
        def __getattr__(self, name):
            attr = getattr(self.wrapped_instance, name)
            if callable(attr):
                return self.measure_method_execution(attr)
            return attr
        
        def measure_method_execution(self, method):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = method(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Execution time of {method.__name__}: {execution_time} seconds")
                return result
            return wrapper
    
    return WrapperClass