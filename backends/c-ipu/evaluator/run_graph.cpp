// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <functional>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>

#if 0
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>

#include <popnn/BatchNorm.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/SpatialSoftMax.hpp>
#include <popnn/codelets.hpp>

#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Gather.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#endif


poplar::Device device;
std::vector<poplar::ComputeSet> compsets;
poplar::program::Sequence prog;

#if __IPU_ARCH_VERSION__ == 2
const int NUM_TILES = 1472;
#else
const int NUM_TILES = 1216;
#endif

namespace
{
    // Return a HW device with the requested number of IPUs.
    // Exception is thrown if no devices with the requested
    // number are available.
    inline poplar::Device getIpuHwDevice(std::size_t numIpus)
    {
        auto dm = poplar::DeviceManager::createDeviceManager();
        auto hwDevices = dm.getDevices(poplar::TargetType::IPU, numIpus);
        if (hwDevices.size() > 0)
        {
            for (auto& d : hwDevices)
            {
                if (d.attach())
                {
                    return std::move(d);
                }
            }
        }
        throw std::runtime_error("No IPU hardware available.");
    }

    // Return an IPU Model device with the requested number of IPUs.
    inline poplar::Device getIpuModelDevice(std::size_t numIpus)
    {
        poplar::IPUModel ipuModel;
        ipuModel.numIPUs = numIpus;
        return ipuModel.createDevice();
    }
}

void place_tensor(poplar::Graph& g, poplar::Tensor& tensor)
{
    return poputil::mapTensorLinearly(g, tensor);

    int num_cells = tensor.numElements(), per_tile, y;
    for (int i = NUM_TILES; i >= 1; --i)
        if (num_cells % i == 0)
        {
            y = i, per_tile = num_cells / i;
            break;
        }
    auto t = tensor.reshape({(size_t)y, (size_t)per_tile});
    for (int i = 0; i < y; ++i)
        g.setTileMapping(t[i], i);
}

inline std::vector<size_t> get_input_offset(const std::vector<size_t> &lo, const std::vector<size_t> &ro, const std::vector<std::vector<int>> &book, int extend) {
    std::vector<size_t> ret;
    for (int i = 0; i < book.size(); ++i) {
        auto &it = book[i];
        if (it[1] < 0 || it[0] == 0)
            ret.push_back(extend ? (it[3] + 1) : (it[2]));
        else if (it[0] > 0)
            ret.push_back(extend ? (it[0] * (ro[it[1]] - 1) + it[3] + 1) : (it[0] * lo[it[1]] + it[2]));
        else
            throw std::runtime_error("Unhandled book case: negative k value.");
    }
    return std::move(ret);
}

inline std::vector<size_t> get_output_offset(const std::vector<int> &part, const std::vector<int> &local_shape, int extend) {
    std::vector<size_t> ret;
    for (int i = 0; i < part.size(); ++i)
        ret.push_back((part[i] + extend) * local_shape[i]);
    return std::move(ret);
}

std::vector<std::vector<int>> compute_iter_parts(const std::vector<int> &nparts) {
    std::vector<int> one(nparts.size());
    std::vector<std::vector<int>> ret{one};
    size_t total = std::accumulate(nparts.begin(), nparts.end(), 1, std::multiplies<int>());
    for (int i = 1; i < total; ++i) {
        int bit = one.size() - 1;
        one[bit]++;
        while (one[bit] >= nparts[bit]) {
            one[bit] = 0;
            one[--bit]++;
        }
        ret.push_back(one);
    }
    return std::move(ret);
}

std::string get_between(const std::string &str, const std::string &begin, const std::string &end) {
    int at = str.find(begin);
    if (at < 0)
        return "";
    at += begin.size();
    int next = str.find(end, at);
    if (next < at)
        return "";
    return str.substr(at, next - at);
}

std::vector<std::string> ssplit(const std::string &str, const std::string &sub) {
    std::vector<std::string> ret;
    int it = 0, next;
    while (next = str.find(sub, it), next >= 0) {
        if (next > it)
            ret.push_back(str.substr(it, next - it));
        it = next + sub.size();
    }
    if (str.size() > it)
        ret.push_back(str.substr(it));
    return std::move(ret);
}


int main(int argc, char** argv)
{
    using namespace poplar;

    device = getenv("EMU") == nullptr ? getIpuHwDevice(1) : getIpuModelDevice(1);
    printf("Ipu Device Id = %d\n", (int)device.getId());

    Graph g(device.getTarget());
    // poplin::addCodelets(g);
    // popnn::addCodelets(g);
    // popops::addCodelets(g);

    std::string path = (argc > 1) ? argv[1] : "my_kernel.cc";

    std::ifstream t(path);
    std::string source((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    t.close();

    auto encoded_params = get_between(source, "// GLOBALS: ", "\n");
    auto params = ssplit(encoded_params, " -> ");
    auto inputs = ssplit(params[0] + ", ", "], "), outputs = ssplit(params[1] + ", ", "], ");

    auto parse_shape = [](const std::string &encoded) -> std::vector<std::size_t> {
        std::vector<std::size_t> shape;
        auto sshape = ssplit(encoded, ", ");
        for (auto &it: sshape)
            shape.push_back(std::atol(it.c_str()));
        return std::move(shape);
    };

    std::unordered_map<std::string, std::vector<float>> host_data;
    std::unordered_map<std::string, poplar::Tensor> feed_dict;
    for (int i = 0; i < inputs.size(); ++i) {
        auto args = ssplit(inputs[i], "[");
        auto shape = parse_shape(args[1]);
        args = ssplit(args[0], ":");
        assert(args[1] == "float32");

        auto &tensor = feed_dict[args[0]];
        // tensor = g.addConstant<T>(dtype, shapes, hptr);
        tensor = g.addVariable(poplar::FLOAT, poplar::ArrayRef<std::size_t>(shape), args[0]);
        place_tensor(g, tensor);

        auto istream = g.addHostToDeviceFIFO("write_" + args[0], FLOAT, tensor.numElements());
        prog.add(poplar::program::Copy(istream, tensor));
        auto &h_data = host_data[args[0]];
        h_data.resize(tensor.numElements());
        for (int c = 0; c < h_data.size(); ++c)
            h_data[c] = (1 + i + c) % 71;

#if 0
        fprintf(stderr, "KernelInfo: Input(%s) = (shape: ", args[0].c_str());
        for (auto &it: shape)
            fprintf(stderr, " %ld,", it);
        fprintf(stderr, ");\n");
#endif
    }

    assert(outputs.size() == 1);
    auto args = ssplit(outputs[0], "[");
    auto shape = parse_shape(args[1]);
    args = ssplit(args[0], ":");
    auto output_name = args[0];

#if 0
    fprintf(stderr, "KernelInfo: Output(%s) = (shape: ", output_name.c_str());
    for (auto &it: shape)
        fprintf(stderr, " %ld,", it);
    fprintf(stderr, ");\n");
#endif
    assert(args[1] == "float32");

    printf("Ipu starts to build..\n");
    poplar::Tensor result = g.addVariable(poplar::FLOAT, poplar::ArrayRef<std::size_t>(shape), output_name);
    std::stringstream codelet;
    codelet << source;
    g.addCodelets(codelet);

    auto compset = g.addComputeSet(__func__);
    prog.add(poplar::program::Execute(compset));

    std::unordered_map<std::string, std::vector<std::vector<int>>> range_book;
    auto property = get_between(source, "\n// Antares Property", "\n");
    property = get_between(property + "\n", ": ", "\n");
    auto tensor_attrs = ssplit(property, ";");
    for (auto& it: tensor_attrs) {
      auto fields = ssplit(it, "/");
      auto &book = range_book[fields[0]];
      book.resize(fields.size() - 1);
      for (int i = 0; i < book.size(); ++i) {
        auto scalers = ssplit(fields[i + 1], ",");
        std::vector<int> one;
        for (auto &s: scalers)
        one.push_back(std::atoi(s.c_str()));
        book[i] = std::move(one);
      }
    }
    std::vector<int> nparts, local_shape;
    for (int it = 0; it < shape.size(); ++it) {
        auto str = get_between(source, "tile_" + std::to_string(it) + "\": [-1, ", "]");
        if (str.size() > 0) {
          auto levels = ssplit(str, ", ");
          int h = std::atoi(levels[0].c_str()), l = std::atoi(levels[1].c_str());
          nparts.push_back(h * l);
	} else
          nparts.push_back(1);
        nparts[it] = shape[it] / nparts[it];
        assert(shape[it] % nparts.back() == 0);
        local_shape.push_back(shape[it] / nparts.back());
    }

    std::vector<std::vector<int>> parts = compute_iter_parts(nparts);
    std::string function_name = "CODELET_" + get_between(source, "\nclass CODELET_", ":");
    auto vector_to_string = [](const std::vector<size_t> &vec) {
        std::string str = "{";
        for (int i = 0; i < vec.size(); ++i)
            str += std::to_string(vec[i]) + ", ";
        return str.substr(0, str.size() - 2) + "}";
    };
    // fprintf(stderr, "[POP_DEBUG] Tile Mapping Information:\n");
    for (int it = 0; it < parts.size(); ++it) {
        poplar::VertexRef v = g.addVertex(compset, function_name);
#if __IPU_ARCH_VERSION__ == 2
        if (g.getTarget().getTargetType() == poplar::TargetType::IPU_MODEL) g.setPerfEstimate(v, 10);
#else
        if (g.getTarget().getTargetType() == poplar::TargetType::IPU_MODEL) g.setCycleEstimate(v, 10);
#endif
        auto lo = get_output_offset(parts[it], local_shape, 0), ro = get_output_offset(parts[it], local_shape, 1);
        int tile_node = it % NUM_TILES;
        // fprintf(stderr, "\n[POP_DEBUG] <Tile Numer = %d>:\n", it);
        // fprintf(stderr, "[POP_DEBUG]     vertex.bindOutputRange(%s.slice(%s, %s));\n", output_name.c_str(), vector_to_string(lo).c_str(), vector_to_string(ro).c_str());
        auto vo = result.slice(lo, ro).flatten();
        g.connect(v[output_name], vo);
        g.setTileMapping(vo, tile_node);
        g.setTileMapping(v, tile_node);
        for (auto& inp: range_book) {
            auto ilo = get_input_offset(lo, ro, inp.second, 0), iro = get_input_offset(lo, ro, inp.second, 1);
            // fprintf(stderr, "[POP_DEBUG]     vertex.bindInputRange(%s.slice(%s, %s));\n", inp.first.c_str(), vector_to_string(ilo).c_str(), vector_to_string(iro).c_str());
            g.connect(v[inp.first], feed_dict[inp.first].slice(ilo, iro).flatten());
        }
        // fprintf(stderr, "[POP_DEBUG]\n");
    }
    fprintf(stderr, "\n");

    size_t out_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    auto ostream = g.addDeviceToHostFIFO("read_out",  FLOAT, out_elements);
    prog.add(program::Copy(result, ostream));
    // prog.add(poplar::program::PrintTensor("Result", out));

    Engine engine(g, prog);
    engine.load(device);

    std::vector<float> h_result(result.numElements());
    engine.connectStream("read_out",  h_result.data());
    for (auto& it: host_data)
        engine.connectStream("write_" + it.first,  it.second.data());

    std::cout << "Running program\n";
    auto run = [&](int runs = 1) -> double {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i)
            engine.run(0);
        auto end = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        return ns * 1e-9 / runs;
    };

    run(1);
    double digest = 0.0;
    for (int i = 0; i < result.numElements(); ++i)
        digest += (i + 1) % 83 * h_result[i];
    printf("- K/0: %.10e\n", digest);

    run(1);
    auto tpr = run(100);
    printf("- TPR: %g\n", tpr);

    if (getenv("PROF"))
        engine.printProfileSummary(std::cerr, {{"showExecutionSteps", "true"}});

    device.detach();
    return EXIT_SUCCESS;
}
