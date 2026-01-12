#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "filter.hh"
#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1-t0).count();
    if (dt == 0) { return 0; }
    return ((n+n+n)*sizeof(float)*1e-9)/(dt*1e-6);
}

void print(const char* name, std::array<duration,5> dt) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names() {
    std::cout << std::setw(19) << "function";
    std::cout << std::setw(20) << "OpenMP";
    std::cout << std::setw(20) << "OpenCL total";
    std::cout << std::setw(20) << "OpenCL copy-in";
    std::cout << std::setw(20) << "OpenCL kernel";
    std::cout << std::setw(20) << "OpenCL copy-out";
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void scan_recursive_helper(OpenCL& opencl, cl::Buffer buffer, int n, int local_size) {
   std::vector<cl::Buffer> block_sums_levels;
    std::vector<int> n_levels;
    
    cl::Buffer current_buffer = buffer;
    int current_n = n;

    // Собираем суммы блоков до тех пор, пока массив не поместится в один рабочий блок
    while (current_n > local_size) {
        int num_groups = (current_n + local_size - 1) / local_size;
        cl::Buffer block_sums(opencl.context, CL_MEM_READ_WRITE, num_groups * sizeof(int));
        
        cl::Kernel k_scan(opencl.program, "scan_inclusive");
        k_scan.setArg(0, current_buffer);
        k_scan.setArg(1, block_sums);
        k_scan.setArg(2, cl::Local(local_size * sizeof(int)));
        k_scan.setArg(3, current_n);
        k_scan.setArg(4, 1); 

        int global_size = num_groups * local_size;
        opencl.queue.enqueueNDRangeKernel(k_scan, cl::NullRange, 
                                          cl::NDRange(global_size), 
                                          cl::NDRange(local_size));

        block_sums_levels.push_back(block_sums);
        n_levels.push_back(current_n);
        
        current_buffer = block_sums;
        current_n = num_groups;
    }

    cl::Kernel k_scan_final(opencl.program, "scan_inclusive");
    k_scan_final.setArg(0, current_buffer);
    k_scan_final.setArg(1, current_buffer);
    k_scan_final.setArg(2, cl::Local(local_size * sizeof(int)));
    k_scan_final.setArg(3, current_n);
    k_scan_final.setArg(4, 0); 

    int final_global = ((current_n + local_size - 1) / local_size) * local_size;
    opencl.queue.enqueueNDRangeKernel(k_scan_final, cl::NullRange, cl::NDRange(final_global), cl::NDRange(local_size));

    // Применяем вычисленные суммы к элементам исходных блоков
    for (int i = block_sums_levels.size() - 1; i >= 0; --i) {
        cl::Buffer& sums = block_sums_levels[i];
        cl::Buffer& target = (i == 0) ? buffer : block_sums_levels[i-1];
        int target_n = n_levels[i];
        
        cl::Kernel k_add(opencl.program, "add_uniform");
        k_add.setArg(0, target);
        k_add.setArg(1, sums);
        k_add.setArg(2, target_n);

        int global_size = ((target_n + local_size - 1) / local_size) * local_size;
        opencl.queue.enqueueNDRangeKernel(k_add, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
    }
}

void profile_filter(int n, OpenCL& opencl) {
    auto input = random_std_vector<float>(n);
    std::vector<float> result;
    std::vector<float> expected_result;
    expected_result.reserve(n);
    auto t0 = clock_type::now();
    filter(input, expected_result, [] (float x) { return x > 0; });
    auto t1 = clock_type::now();
    cl::Buffer d_input(opencl.queue, begin(input), end(input), true);
    cl::Buffer d_mask(opencl.context, CL_MEM_READ_WRITE, n * sizeof(float));
    cl::Buffer d_indices(opencl.context, CL_MEM_READ_WRITE, n * sizeof(float));
    cl::Buffer d_output(opencl.context, CL_MEM_READ_WRITE, n * sizeof(float));
    opencl.queue.finish();
    auto t2 = clock_type::now();
    // 1. MAP
    cl::Kernel k_map(opencl.program, "filter_map");
    k_map.setArg(0, d_input);
    k_map.setArg(1, d_mask);
    k_map.setArg(2, n);
    
    int local_size = 256;
    int num_groups = (n + local_size - 1) / local_size;
    int global_size = num_groups * local_size;
    
    opencl.queue.enqueueNDRangeKernel(k_map, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));

    // 2. SCAN 
    opencl.queue.enqueueCopyBuffer(d_mask, d_indices, 0, 0, n * sizeof(float));
    scan_recursive_helper(opencl, d_indices, n, local_size);
    float count_f;
    opencl.queue.enqueueReadBuffer(d_indices, CL_TRUE, (n - 1) * sizeof(float), sizeof(float), &count_f);
    int count = (int)count_f;

    auto t3 = clock_type::now();

    // 3. SCATTER
    if (count > 0) {
        cl::Kernel k_scatter(opencl.program, "filter_scatter");
        k_scatter.setArg(0, d_input);
        k_scatter.setArg(1, d_indices); 
        k_scatter.setArg(2, d_output);  
        k_scatter.setArg(3, n);   
        
        opencl.queue.enqueueNDRangeKernel(k_scatter, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
        
        result.resize(count);
        opencl.queue.enqueueReadBuffer(d_output, CL_TRUE, 0, count * sizeof(float), result.data());
    } else {
        result.clear();
    }
    
    auto t4 = clock_type::now();
    verify_vector(expected_result, result);
    print("filter", {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3});
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_filter(1024*1024, opencl);
}

const std::string src = R"(
kernel void filter_map(global const float* input,
                       global float* mask,
                       int n) {
    int global_id = get_global_id(0);
    if (global_id < n) {
        mask[global_id] = (input[global_id] > 0) ? 1.0f : 0.0f;
    }
}

kernel void filter_scatter(global const float* input,
                           global const float* indices,
                           global float* output,
                           int n) {
    int global_id = get_global_id(0);
    
    if (global_id < n) {
        float val = input[global_id];
        
        if (val > 0) { 
            int idx = (int)indices[global_id] - 1;
            output[idx] = val;
        }
    }
}

kernel void scan_inclusive(global float* a,
                           global float* sums,
                           local float* data,
                           int n,
                           int store_sum) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);

    if (global_id < n) {
        data[local_id] = a[global_id];
    } else {
        data[local_id] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = data[local_id];

    // Скан в локальной памяти
    for (int offset=1; offset<local_size; offset *= 2) {
        float other = 0.0f;
        if (local_id >= offset) {
            other = data[local_id - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (local_id >= offset) {
            sum += other;
        }
        data[local_id] = sum; 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_id < n) {
        a[global_id] = data[local_id];
    }

    if (store_sum && local_id == local_size - 1) {
        sums[group_id] = data[local_id];
    }
}

kernel void add_uniform(global float* data,
                        global const float* sums,
                        int n) {
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    if (group_id > 0 && global_id < n) {
        data[global_id] += sums[group_id - 1];
    }
}
)";

int main() {
    try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }
                cl::Platform platform;
        bool found = false;

        // Fixing the device boot queue
        // First - NVIDIA
        for (const auto& p : platforms) {
            std::string name = p.getInfo<CL_PLATFORM_NAME>();
            if (name.find("NVIDIA") != std::string::npos) {
                platform = p;
                found = true;
                break;
            }
        }
        
        // found = false;
        // Second - Rusticl (for AMD)
        if (!found) {
            for (const auto& p : platforms) {
                std::string name = p.getInfo<CL_PLATFORM_NAME>();
                if (name.find("rusticl") != std::string::npos) { 
                    platform = p;
                    found = true;
                    break;
                }
            }
        }

        // fallback - zero device
        if (!found) {
            platform = platforms[0];
            std::cerr << "Warning: Using default platform (might be broken Clover)\n";
        }

        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        // compile the programme
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
