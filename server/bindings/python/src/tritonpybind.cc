#include <pybind11/functional.h>
#include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>

#include <deque>
#include <iterator>
#include <string>
#include <utility>

#include "triton/developer_tools/server_wrapper.h"
namespace tds = triton::developer_tools::server;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define FAIL(MSG)                                 \
  do {                                            \
    std::cerr << "error: " << (MSG) << std::endl; \
    exit(1);                                      \
  } while (false)


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


namespace py = pybind11;
using namespace pybind11::literals;

std::unique_ptr<tds::TritonServer> server_;
std::string model_name_;
// concurrent_queue<std::unique_ptr<tds::InferResult>> result_queue_;
concurrent_queue<int> result_queue_;


// Create a server with a model
static void
ceeate_server(const std::string& model_name)
{
  tds::ServerOptions options(
      {"/home/giqbal/Source/triton/developer_tools/server/examples/models"});
  options.backend_dir_ =
      "/home/giqbal/Source/triton/developer_tools/server/examples/backends";
  options.model_control_mode_ = tds::ModelControlMode::EXPLICIT;
  options.logging_.verbose_ = tds::LoggingOptions::VerboseLevel::OFF;

  auto trace = std::make_shared<tds::Trace>(
      "/home/giqbal/Source/triton/triton_trace.log", tds::Trace::Level::TENSORS,
      1, -1, 1);
  // const std::string& file, const Level& level, const uint32_t rate,
  // const int32_t count, const uint32_t log_frequency);
  // options.trace_ = trace;


  server_ = tds::TritonServer::Create(options);

  server_->LoadModel(model_name);

  std::set<std::string> loaded_models = server_->LoadedModels();
  if (loaded_models.find(model_name) == loaded_models.end()) {
    FAIL("Model '" + model_name + "' is not found.");
  }

  model_name_ = model_name;
}

static py::dict
create_response(std::unique_ptr<tds::InferResult>& result)
{
  py::dict response;
  for (auto n : result->OutputNames()) {
    py::str key = n;
    std::shared_ptr<tds::Tensor> out_tensor = result->Output(n);
    std::vector<ssize_t> result_shape;
    for (auto s : out_tensor->shape_) {
      result_shape.push_back(s);
    }
    // std::vector<ssize_t> result_strides;
    // using ShapeContainer = detail::any_container<ssize_t>;
    // using StridesContainer = detail::any_container<ssize_t>;

    switch (out_tensor->data_type_) {
      case tds::DataType::FP32:
        const float* output_data =
            reinterpret_cast<const float*>(out_tensor->buffer_);
        auto arr = py::array_t<float>(result_shape, output_data);
        response[key] = arr;
        std::cout << "Created Response Array" << std::endl;
        break;
        // case tds::DataType::INT32:
    }
  }
  // m.def("get_dict", []() { return py::dict("key"_a = "value"); });
  return response;
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


static std::unique_ptr<tds::InferRequest>
create_request(const py::dict& dict)
{
  std::cout << "pybind: create_request:  Thread: " << std::this_thread::get_id()
            << std::endl;

  auto request = tds::InferRequest::Create(tds::InferOptions(model_name_));
  for (auto item : dict) {
    const auto& key = item.first;
    const auto& value = item.second;
    if (!py::isinstance<py::str>(key)) {
      throw py::type_error("Expected string");
    }
    if (!py::isinstance<py::array>(value)) {
      throw py::type_error("Expected array");
    }

    std::string key_str = py::str(key);

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

      auto request_buffer = new float[buffer.size];
      for (auto i = 0; i < buffer.size; ++i) {
        request_buffer[i] = buffer_start[i];
      }

      request->AddInput(
          key_str,
          // buffer_start, buffer_end,
          &request_buffer[0], &request_buffer[buffer.size], tds::DataType::FP32,
          shape, tds::MemoryType::CPU, 0);

      // auto arr_unchecked = arr.unchecked();
      // auto buffer = arr.request();
      // const auto* arr_start = arr_unchecked.data();
      // const auto* arr_end = arr_unchecked.
    } else if (py::isinstance<py::array_t<std::double_t>>(value)) {
      auto arr = py::array_t<std::double_t>::ensure(value);
      const auto& dtype = arr.dtype();
      py::print("double_t dtype: {}, {}, {}"_s.format(
          dtype, dtype.kind(), dtype.itemsize()));
    }

    // py::array arr = py::array::ensure(value);
    // py::print("array: {}, {}"_s.format(arr.ndim(), arr.nbytes()));
    // const auto& dtype = arr.dtype();
    // py::print(
    //     "dtype: {}, {}, {}"_s.format(dtype, dtype.kind(), dtype.itemsize()));
  }
  return request;
}

// Execute request
static py::dict
predict(const py::dict& dict)
{
  auto request = create_request(dict);
  auto future = server_->AsyncInfer(*request);
  auto result = future.get();
  auto response = create_response(result);
  return response;
}

// static void
// pybind_callback_fn(
//     std::unique_ptr<tds::InferResult> result, const py::function& f)
// {
//   std::cout << "PyBind Callback" << std::endl;

//   if (result->HasError()) {
//     FAIL(result->ErrorMsg());
//   }
//   std::string name = result->ModelName();
//   std::string version = result->ModelVersion();
//   std::string id = result->Id();
//   std::cout << "Ran inference on model '" << name << "', version '" <<
//   version
//             << "', with request ID '" << id << "'\n";

//   for (auto n : result->OutputNames()) {
//     std::cout << "OutputName: " << n << std::endl;
//     std::shared_ptr<tds::Tensor> out_tensor = result->Output(n);
//     std::cout << "Bytes Size: " << out_tensor->byte_size_ << std::endl;
//     std::cout << "Data Type: " << tds::DataTypeString(out_tensor->data_type_)
//               << std::endl;
//     std::cout << "Memory Type: " <<
//     static_cast<int>(out_tensor->memory_type_)
//               << std::endl;
//     for (auto s : out_tensor->shape_) {
//       std::cout << "- Shape: " << s << std::endl;
//     }
//     const float* output_data =
//         reinterpret_cast<const float*>(out_tensor->buffer_);
//     for (int i = 0; i < 3; ++i) {
//       std::cout << "- Result " << i << "=" << output_data[i] << std::endl;
//     }
//   }
//   std::cout << result->DebugString() << std::endl;

//   std::cout << "PyBind Callback Invoking" << std::endl;
//   {
//     py::gil_scoped_acquire acquire;
//     auto response = create_response(result);
//     py::print("PyBindCall - Calling Python");
//     py::dict empty;
//     f(empty);
//     py::print("PyBindCall - Called Python");
//   }
// }

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


static void
predict_with_callback(const py::dict& dict, const py::function& f)
{
  std::cout << "Thread: triton request: " << std::this_thread::get_id()
            << std::endl;

  if (!callback_thread) {
    // callback_thread = std::make_unique<std::thread>(callback_thread_fn);
    // python_callback = f;
  }

  // auto thread_fn = [f = std::move(f)](int x) {
  //   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  //   std::cout << "Thread: pybind callback: " << std::this_thread::get_id()
  //             << std::endl;
  //   py::gil_scoped_acquire acquire;
  //   py::dict empty;
  //   empty["blah"] = x;
  //   f(empty);
  // };
  // auto t = std::thread(thread_fn, 123);
  // t.detach();

  auto request = create_request(dict);

  auto callback_fn =  // [](std::unique_ptr<tds::InferResult> result) {
      [f = std::move(f)](std::unique_ptr<tds::InferResult> result) {
        // [f = std::move(f)]() {
        // pybind_callback_fn(std::move(result), std::move(f));
        std::cout << "Thread: triton callback: " << std::this_thread::get_id()
                  << std::endl;
        // result_queue_.push(1);
        {
          py::gil_scoped_acquire acquire{};
          py::dict empty;
          empty["blah"] = "blah";
          f(empty);
        }
      };

  py::gil_scoped_release release;
  server_->AsyncInferWithCallback(*request, std::move(callback_fn));
  callback_request = std::move(request);


  // py::dict empty;
  // f(empty);

  // server_->AsyncInferWithCallback(
  //     *request, [&f](std::unique_ptr<tds::InferResult> result) {
  //       pybind_callback_fn(std::move(result), f);
  //     });


  // auto future = server_->AsyncInfer(*request);
  // auto result = future.get();
  // auto response = create_response(std::move(result));
  // f(response);


  // server_->AsyncInferWithCallback(
  //     *request, [f](std::unique_ptr<tds::InferResult> result) {
  //       pybind_callback_fn(std::move(result), f);
  //     });

  // auto request = create_request(dict);
  // auto result_future = server_->AsyncInfer(*request);
  // server_->AsyncInferWithCallback(*request, pybind_callback_fn);

  // Call 'AsyncInfer' function to run inference.
  // auto result_future = server_->AsyncInfer(*request);
  // // Get the infer result and check the result.
  // auto result = result_future.get();
  // pybind_callback_fn(std::move(result));


  // sleep(1);
}


// Shutdown server


// py::print("key: {}, value={}"_s.format(key, value));

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

    // if (!py::buffer_info(value).check()) {
    //   throw py::type_error("Expected np array");
    // }
  }

  //  if (buf1.ndim != 1 || buf2.ndim != 1)
  //       throw std::runtime_error("Number of dimensions must be one");

  //  if (buf1.shape[0] != buf2.shape[0])
  //       throw std::runtime_error("Input shapes must match");

  //   auto result = py::array(py::buffer_info(
  //       nullptr,            /* Pointer to data (nullptr -> ask NumPy to
  //       allocate!) */ sizeof(double),     /* Size of one item */
  //       py::format_descriptor<double>::value, /* Buffer format */
  //       buf1.ndim,          /* How many dimensions? */
  //       { buf1.shape[0] },  /* Number of elements for each dimension */
  //       { sizeof(double) }  /* Strides for each dimension */
  //   ));

  //   auto buf3 = result.request();

  //   double *ptr1 = (double *) buf1.ptr,
  //          *ptr2 = (double *) buf2.ptr,
  //          *ptr3 = (double *) buf3.ptr;

  //   for (size_t idx = 0; idx < buf1.shape[0]; idx++)
  //       ptr3[idx] = ptr1[idx] + ptr2[idx];

  //   return result;
}


PYBIND11_MODULE(tritonpybind, m)
{
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

  m.def("create_server", &ceeate_server, "Create a triton server");
  m.def("predict", &predict, "Run a prediction");
  m.def("predict_with_callback", &predict_with_callback, "Run a prediction");

  m.def("sum_dict", &sum_dict, "Sum a dict of NP arrays");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}