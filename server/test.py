import numpy as np
import site
import faulthandler
import gc
import threading
import time

faulthandler.enable()

pip_package = "/home/giqbal/Source/triton/build/bindings/python/"
site.addsitedir(pip_package)

import triton


def print_prediction_result(result):
    for k, v in result.items():
        print(f"Result: {k}: {v[:5]}")


executor = triton.Executor("/home/giqbal/Source/triton/developer_tools/server/examples/models", "/home/giqbal/Source/triton/developer_tools/server/examples/backends")
print(executor)

data = np.zeros([3, 224, 224]).astype(np.float32)
query = {
    "data_0": data
}

session = executor.create_session("densenet_onnx")
result = session.predict(query)
print_prediction_result(result)

ev = threading.Event()

def callback_fn(result):
    print(f"Python Callback: {threading.current_thread().ident}")
    print_prediction_result(result)
    ev.set()

print(f"Thread Request: {threading.current_thread().ident}")
session.predict_with_callback(query, callback_fn)
ev.wait()

executor = None

time.sleep(0.5)
print("Bye")

"""

# model_session = triton.session(model_path)
# model_session.predict()

print(dir(tritonpybind))
# print(tritonpybind.add(1,2))

# arr = np.ndarray([1, 2, 3])
# print(arr)
# d = dict({"one":arr})          
# tritonpybind.sum_dict(d)
tritonpybind.create_server("densenet_onnx")


# print(f"Before: {gc.get_referents(data)}, {len(gc.get_referrers(data))}")
# result = tritonpybind.predict(query)
# print("Result1 PreGC:", result.items())
# gc.collect()
# print("Result2 PostGC:",result.items())
# result = tritonpybind.predict(query)
# print("Result3 PostGC:",result.items())

ev = threading.Event()

def callback_fn(result):
    print(f"Thread Callback: {threading.current_thread().ident}")
    print("Result3 PostGC:",result.items())
    ev.set()

print(f"Thread Request: {threading.current_thread().ident}")
tritonpybind.predict_with_callback(query, callback_fn)
print(f"Thread Post-Request: {threading.current_thread().ident}")

ev.wait()
# time.sleep(5.0)
print("Exiting")
# print(f"After: {gc.get_referents(data)}, {len(gc.get_referrers(data))}")

"""