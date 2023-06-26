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