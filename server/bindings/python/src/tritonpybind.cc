// #include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <unistd.h>

#include <deque>
#include <iterator>
#include <string>
#include <utility>

#include "triton/developer_tools/server_wrapper.h"
namespace tds = triton::developer_tools::server;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;


#if 0
template <typename T>
class concurrent_queue {
  std::deque<T> _deque;
  std::mutex _mutex;
  std::condition_variable _cond;

 public:
  concurrent_queue() {}
  concurrent_queue(const concurrent_queue& other) = delete;

  size_t size() const { return _deque.size(); }

  void push(const T&& item)
  {
    std::lock_guard<std::mutex> guard(_mutex);
    _deque.push_back(item);
    _cond.notify_one();
  }
  std::vector<T> drain()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cond.wait(lock, [this] { return !_deque.empty(); });
    std::vector<T> result(
        std::make_move_iterator(_deque.begin()),
        std::make_move_iterator(_deque.end()));
    _deque.erase(_deque.begin(), _deque.end());
    return result;
  }
};


std::unique_ptr<tds::TritonServer> server_;
std::string model_name_;
// concurrent_queue<std::unique_ptr<tds::InferResult>> result_queue_;
concurrent_queue<int> result_queue_;

static void
pybind_callback_fn(
    std::unique_ptr<tds::InferResult> result, const py::function& f)
{
  std::cout << "PyBind Callback" << std::endl;

  if (result->HasError()) {
    FAIL(result->ErrorMsg());
  }
  std::string name = result->ModelName();
  std::string version = result->ModelVersion();
  std::string id = result->Id();
  std::cout << "Ran inference on model '" << name << "', version '" <<
  version
            << "', with request ID '" << id << "'\n";

  for (auto n : result->OutputNames()) {
    std::cout << "OutputName: " << n << std::endl;
    std::shared_ptr<tds::Tensor> out_tensor = result->Output(n);
    std::cout << "Bytes Size: " << out_tensor->byte_size_ << std::endl;
    std::cout << "Data Type: " << tds::DataTypeString(out_tensor->data_type_)
              << std::endl;
    std::cout << "Memory Type: " <<
    static_cast<int>(out_tensor->memory_type_)
              << std::endl;
    for (auto s : out_tensor->shape_) {
      std::cout << "- Shape: " << s << std::endl;
    }
    const float* output_data =
        reinterpret_cast<const float*>(out_tensor->buffer_);
    for (int i = 0; i < 3; ++i) {
      std::cout << "- Result " << i << "=" << output_data[i] << std::endl;
    }
  }
  std::cout << result->DebugString() << std::endl;

  std::cout << "PyBind Callback Invoking" << std::endl;
  {
    py::gil_scoped_acquire acquire;
    auto response = create_response(result);
    py::print("PyBindCall - Calling Python");
    py::dict empty;
    f(empty);
    py::print("PyBindCall - Called Python");
  }
}

// static py::function&
static std::function<void(py::dict)> python_callback;

static void
callback_thread_fn()
{
  while (1) {
    auto results = result_queue_.drain();
    std::cout << "Thread: callback: " << std::this_thread::get_id()
              << std::endl;
    {
      py::gil_scoped_acquire acquire{};
      py::dict empty;
      empty["blah"] = "blah";
      python_callback(empty);
    }
  }
}

static std::unique_ptr<std::thread> callback_thread;
static std::unique_ptr<tds::InferRequest> callback_request;

void
sum_dict(const py::dict& dict)
{
  std::unique_ptr<py::array> result;

  for (auto item : dict) {
    const auto& key = item.first;
    const auto& value = item.second;
    if (!py::isinstance<py::str>(key)) {
      throw py::type_error("Expected string");
    }
    if (!py::isinstance<py::array>(value)) {
      throw py::type_error("Expected array");
    }
    py::array arr = py::array::ensure(value);
    py::print("array: {}, {}"_s.format(arr.ndim(), arr.nbytes()));
    const auto& dtype = arr.dtype();
    py::print(
        "dtype: {}, {}, {}"_s.format(dtype, dtype.kind(), dtype.itemsize()));

    if (!result) {
      // arr.dtype().
      // auto buffer_info = py::buffer_info(nullptr, arr.itemsize(),
      // How does
      // result = std::make_unique(py::array(py::buffer_info(
    }
  }
}

// BOOL,
// UINT8,
// UINT16,
// UINT32,
// UINT64,
// INT8,
// INT16,
// INT32,
// INT64,
// FP16,
// FP32,
// FP64,
// BYTES,
// BF16

#endif

static std::unique_ptr<tds::InferRequest>
create_inference_request(const std::string& model_name, const py::dict& dict)
{
  std::cout << "PyBind: create_inference_request: " << model_name
            << ", ThreadId: " << std::this_thread::get_id() << std::endl;

  auto request = tds::InferRequest::Create(tds::InferOptions(model_name));
  for (auto item : dict) {
    const auto& key = item.first;
    const auto& value = item.second;
    if (!py::isinstance<py::str>(key)) {
      throw py::type_error("Expected string");
    }
    std::string key_str = py::str(key);

    if (!py::isinstance<py::array>(value)) {
      throw py::type_error("Expected array");
    }

    if (py::isinstance<py::array_t<std::int32_t>>(value)) {
      auto arr = py::array_t<std::int32_t>::ensure(value);
      const auto& dtype = arr.dtype();
      py::print("int32_t dtype: {}, {}, {}"_s.format(
          dtype, dtype.kind(), dtype.itemsize()));

    } else if (py::isinstance<py::array_t<std::float_t>>(value)) {
      auto arr = py::array_t<std::float_t>::ensure(value);
      const auto& dtype = arr.dtype();
      py::print("float_t dtype: {}, {}, {}"_s.format(
          dtype, dtype.kind(), dtype.itemsize()));

      auto buffer = arr.request();
      std::vector<int64_t> shape;
      shape.reserve(buffer.ndim);
      for (auto n = 0; n < buffer.ndim; ++n) {
        shape.push_back(static_cast<int64_t>(buffer.shape[n]));
        py::print("float_t shape: {}"_s.format(buffer.shape[n]));
      }

      auto buffer_size = (buffer.itemsize * buffer.size);
      py::print("float_t buffer: {}, {}, {}"_s.format(
          buffer.itemsize, buffer.size, buffer_size));

      auto buffer_start = reinterpret_cast<float*>(buffer.ptr);
      auto buffer_end = buffer_start + buffer_size;

      // auto request_buffer = new float[buffer.size];
      // for (auto i = 0; i < buffer.size; ++i) {
      //   request_buffer[i] = buffer_start[i];
      // }
      //  &request_buffer[0], &request_buffer[buffer.size]

      request->AddInput(
          key_str, buffer_start, buffer_end, tds::DataType::FP32, shape,
          tds::MemoryType::CPU, 0);

    } else if (py::isinstance<py::array_t<std::double_t>>(value)) {
      auto arr = py::array_t<std::double_t>::ensure(value);
      const auto& dtype = arr.dtype();
      py::print("double_t dtype: {}, {}, {}"_s.format(
          dtype, dtype.kind(), dtype.itemsize()));
    }
  }
  return request;
}

static py::dict
create_inference_response(std::unique_ptr<tds::InferResult>& result)
{
  std::cout << "PyBind: create_inference_response: " << result->ModelName()
            << ", ThreadId: " << std::this_thread::get_id() << std::endl;
  py::dict response;
  for (auto n : result->OutputNames()) {
    std::shared_ptr<tds::Tensor> out_tensor = result->Output(n);
    std::vector<ssize_t> result_shape;
    for (auto s : out_tensor->shape_) {
      result_shape.push_back(s);
    }
    py::str key = n;
    switch (out_tensor->data_type_) {
      case tds::DataType::FP32:
        const float* output_data =
            reinterpret_cast<const float*>(out_tensor->buffer_);
        // TODO@gaz: Is this a copy?
        auto arr = py::array_t<float>(result_shape, output_data);
        response[key] = arr;
        break;
    }
  }
  return response;
}


class TritonModelSession {
  std::string model_name_;
  std::shared_ptr<tds::TritonServer> server_;

 public:
  TritonModelSession(
      std::shared_ptr<tds::TritonServer> server_, const std::string& model_name)
  {
    std::cout << "PyBind: TritonModelSession: Constructing: " << model_name
              << std::endl;
    server_->LoadModel(model_name);
    this->model_name_ = model_name;
    this->server_ = server_;
  }

  ~TritonModelSession()
  {
    std::cout << "PyBind: TritonModelSession: Destructed: " << this->model_name_
              << std::endl;
  }

  py::dict predict(const py::dict& query)
  {
    std::cout << "PyBind: TritonModelSession.predict: " << this->model_name_
              << ", ThreadId: " << std::this_thread::get_id() << std::endl;

    auto request = create_inference_request(this->model_name_, query);
    auto future = server_->AsyncInfer(*request);
    auto result = future.get();
    auto response = create_inference_response(result);
    return response;
  }

  void predict_with_callback(
      const py::dict& query, const py::function& callback_fn)
  {
    std::cout << "PyBind: TritonModelSession.predict_with_callback: "
              << this->model_name_
              << ", ThreadId: " << std::this_thread::get_id() << std::endl;

    std::shared_ptr<tds::InferRequest> request =
        create_inference_request(this->model_name_, query);

    auto inference_callback_fn =
        [callback_fn = std::move(callback_fn),
         request](std::unique_ptr<tds::InferResult> result) {
          std::cout
              << "PyBind: "
                 "TritonModelSession.predict_with_callback.inference_callback: "
              << result->ModelName()
              << ", ThreadId: " << std::this_thread::get_id() << std::endl;
          {
            py::gil_scoped_acquire acquire;
            auto response = create_inference_response(result);
            callback_fn(response);
          }
        };

    py::gil_scoped_release release;
    server_->AsyncInferWithCallback(*request, inference_callback_fn);
  }
};


class TritonModelExecutor {
  std::shared_ptr<tds::TritonServer> server_;

 public:
  TritonModelExecutor(
      const std::string& model_repo_dir, const std::string& backend_dir)
  {
    char current_path[2048];
    ::getcwd(current_path, sizeof(current_path));
    std::cout << "PyBind: TritonModelExecutor: Constructing at: "
              << current_path << std::endl;

    tds::ServerOptions options({model_repo_dir});
    options.backend_dir_ = backend_dir;
    options.model_control_mode_ = tds::ModelControlMode::EXPLICIT;
    options.logging_.verbose_ = tds::LoggingOptions::VerboseLevel::OFF;
    this->server_ = tds::TritonServer::Create(options);
  }

  ~TritonModelExecutor()
  {
    this->server_.reset();
    std::cout << "PyBind: TritonModelExecutor: Destructed" << std::endl;
  }

  std::unique_ptr<TritonModelSession> create_session(
      const std::string& model_name)
  {
    return std::make_unique<TritonModelSession>(this->server_, model_name);
  }

  // Not copyable or movable
  TritonModelExecutor(const TritonModelExecutor&) = delete;
  TritonModelExecutor& operator=(const TritonModelExecutor&) = delete;
};


PYBIND11_MODULE(triton, m)
{
  py::class_<TritonModelSession>(m, "Session")
      .def("predict", &TritonModelSession::predict)
      .def("predict_with_callback", &TritonModelSession::predict_with_callback);


  py::class_<TritonModelExecutor>(m, "Executor")
      .def(py::init<const std::string&, const std::string&>())
      .def("create_session", &TritonModelExecutor::create_session);

  // m.def("create_server", &ceeate_server, "Create a triton server");
  // m.def("predict", &predict, "Run a prediction and wait for its result");
  // m.def("predict_with_callback", &predict_with_callback, "Run a prediction");
  // m.def("predict_async", &predict_with_callback, "Run a prediction");

  // Predict where the callbacks are serialized on a different thread?

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}