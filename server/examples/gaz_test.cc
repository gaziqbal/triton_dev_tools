// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <unistd.h>

#include <iostream>
#include <iterator>
#include <string>

#include "triton/developer_tools/server_wrapper.h"

namespace tds = triton::developer_tools::server;

namespace {

#define FAIL(MSG)                                 \
  do {                                            \
    std::cerr << "error: " << (MSG) << std::endl; \
    exit(1);                                      \
  } while (false)

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v Enable verbose logging" << std::endl;

  exit(1);
}

void
CompareResult(
    const std::vector<std::string>& input0_data,
    const std::vector<std::string>& input1_data,
    const std::vector<std::string>& result0_data,
    const std::vector<std::string>& result1_data,
    const std::vector<int32_t>& expected_sum,
    const std::vector<int32_t>& expected_diff)
{
  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data[i] << " + " << input0_data[i] << " = "
              << result0_data[i] << std::endl;
    std::cout << input0_data[i] << " - " << input1_data[i] << " = "
              << result1_data[i] << std::endl;

    if (expected_sum[i] != std::stoi(result0_data[i])) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if (expected_diff[i] != std::stoi(result1_data[i])) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }
}

void
Check(
    std::shared_ptr<tds::Tensor>& output0,
    std::shared_ptr<tds::Tensor>& output1,
    const std::vector<std::string>& input0_data,
    const std::vector<std::string>& input1_data,
    const std::string& output0_name, const std::string& output1_name,
    const std::vector<std::string>& result0_data,
    const std::vector<std::string>& result1_data,
    const std::vector<int32_t>& expected_sum,
    const std::vector<int32_t>& expected_diff)
{
  for (auto& output :
       {std::make_pair(output0_name, output0),
        std::make_pair(output1_name, output1)}) {
    if ((output.second->shape_.size() != 1) ||
        (output.second->shape_[0] != 16)) {
      std::cerr << "error: received incorrect shapes for " << output.first
                << std::endl;
      exit(1);
    }

    if (output.second->data_type_ != tds::DataType::BYTES) {
      FAIL(
          "unexpected datatype '" +
          std::string(DataTypeString(output.second->data_type_)) + "' for '" +
          output.first + "'");
    }

    if (output.second->memory_type_ != tds::MemoryType::CPU) {
      FAIL(
          "unexpected memory type, expected to be allocated in CPU, got " +
          std::string(MemoryTypeString(output.second->memory_type_)) + ", id " +
          std::to_string(output.second->memory_type_id_) + " for " +
          output.first);
    }
  }

  if (result0_data.size() != 16) {
    std::cerr << "error: received incorrect number of strings for OUTPUT0: "
              << result0_data.size() << std::endl;
  }
  if (result1_data.size() != 16) {
    std::cerr << "error: received incorrect number of strings for OUTPUT1: "
              << result1_data.size() << std::endl;
  }

  CompareResult(
      input0_data, input1_data, result0_data, result1_data, expected_sum,
      expected_diff);
}

}  // namespace

int
main(int argc, char** argv)
{
  int verbose_level = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:H:")) != -1) {
    switch (opt) {
      case 'v':
        verbose_level = 1;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }
  try {
    // Use 'ServerOptions' object to initialize TritonServer.
    tds::ServerOptions options(
        {"/home/giqbal/Source/triton/developer_tools/server/examples/models"});
    options.backend_dir_ =
        "/home/giqbal/Source/triton/developer_tools/server/examples/backends";
    options.model_control_mode_ = tds::ModelControlMode::EXPLICIT;
    options.logging_.verbose_ = tds::LoggingOptions::VerboseLevel::MAX;
    // options.logging_.verbose_ =
    //     tds::LoggingOptions::VerboseLevel(verbose_level);
    auto server = tds::TritonServer::Create(options);

    // We use a simple model that takes 2 input tensors of 16 strings
    // each and returns 2 output tensors of 16 strings each. The input
    // strings must represent integers. One output tensor is the
    // element-wise sum of the inputs and one output is the element-wise
    // difference.
    std::string model_name = "densenet_onnx";
    server->LoadModel(model_name);

    // Use 'LoadedModels' function to check if the model we need is loaded.
    std::set<std::string> loaded_models = server->LoadedModels();
    if (loaded_models.find(model_name) == loaded_models.end()) {
      FAIL("Model '" + model_name + "' is not found.");
    }

    auto metadata = server->ModelMetadata(model_name);
    std::cout << "Model Metadata: " << metadata << std::endl;

    auto request = tds::InferRequest::Create(tds::InferOptions(model_name));

    size_t input_size = 3 * 224 * 224;
    auto input_data = new float[input_size];
    for (size_t i = 0; i < input_size; ++i) {
      input_data[i] = 0.0f;
    }

    std::cout << "Input Size: " << input_size * sizeof(float) << std::endl;

    // std::vector<float> input_data(input_size, 0.0f);
    std::vector<int64_t> shape{3, 224, 224};

    request->AddInput(
        "data_0", &input_data[0], &input_data[input_size],
        // input_data.begin(),
        // input_data.end(),
        tds::DataType::FP32, shape, tds::MemoryType::CPU, 0);


    // // Indicate that we want both output tensors calculated and returned
    // // for the inference request. These calls are optional, if no
    // // output(s) are specifically requested then all outputs defined by
    // // the model will be calculated and returned.
    // request->AddRequestedOutput("OUTPUT0");
    // request->AddRequestedOutput("OUTPUT1");

    // Call 'AsyncInfer' function to run inference.
    auto result_future = server->AsyncInfer(*request);

    // Get the infer result and check the result.
    auto result = result_future.get();
    if (result->HasError()) {
      FAIL(result->ErrorMsg());
    }
    std::string name = result->ModelName();
    std::string version = result->ModelVersion();
    std::string id = result->Id();
    std::cout << "Ran inference on model '" << name << "', version '" << version
              << "', with request ID '" << id << "'\n";

    for (auto n : result->OutputNames()) {
      std::cout << "OutputName: " << n << std::endl;
      std::shared_ptr<tds::Tensor> out_tensor = result->Output(n);
      std::cout << "Bytes Size: " << out_tensor->byte_size_ << std::endl;
      std::cout << "Data Type: " << tds::DataTypeString(out_tensor->data_type_)
                << std::endl;
      std::cout << "Memory Type: " << static_cast<int>(out_tensor->memory_type_)
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
  }
  catch (const tds::TritonException& ex) {
    std::cerr << "Error: " << ex.what();
    exit(1);
  }

  return 0;
}
