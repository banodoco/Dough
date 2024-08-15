from functools import wraps
import json
import os
import time
import portalocker
import streamlit as st
from streamlit import runtime


def count_calls(cls):
    class Wrapper(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.call_counts = {}
            self.total_count = 0

        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if callable(attr) and name not in ["__getattribute__", "call_counts", "total_count"]:
                if name not in self.call_counts:
                    self.call_counts[name] = 0

                def wrapped_method(*args, **kwargs):
                    self.call_counts[name] += 1
                    self.total_count += 1
                    return attr(*args, **kwargs)

                return wrapped_method

            return attr

    return Wrapper


def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"{args[1] if args and len(args) >= 2 else kwargs['url']} took {execution_time:.4f} seconds to execute."
        )
        return result

    return wrapper


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


def session_state_attributes(default_value_cls):
    def decorator(cls):
        original_getattr = cls.__getattribute__
        original_setattr = cls.__setattr__

        def custom_attr(self, attr):
            if hasattr(default_value_cls, attr):
                key = f"{self.uuid}_{attr}"
                if not (key in st.session_state and st.session_state[key]):
                    st.session_state[key] = getattr(default_value_cls, attr)

                return st.session_state[key] if runtime.exists() else getattr(default_value_cls, attr)
            else:
                return original_getattr(self, attr)

        def custom_setattr(self, attr, value):
            if hasattr(default_value_cls, attr):
                key = f"{self.uuid}_{attr}"
                st.session_state[key] = value
            else:
                original_setattr(self, attr, value)

        cls.__getattribute__ = custom_attr
        cls.__setattr__ = custom_setattr
        return cls

    return decorator


def with_refresh_lock(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        update_refresh_lock(True)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("Error occured while processing ", str(e))
        finally:
            update_refresh_lock(False)

    return wrapper


def update_refresh_lock(status=False):
    from utils.constants import REFRESH_LOCK_FILE

    status = "locked" if status else "unlocked"
    lock_file = REFRESH_LOCK_FILE
    with portalocker.Lock(lock_file, "w") as lock_file_handle:
        json.dump(
            {"status": status, "last_action_time": time.time(), "process_id": os.getpid()},
            lock_file_handle,
        )
