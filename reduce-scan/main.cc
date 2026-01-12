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
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

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

void print(const char* name, std::array<duration,5> dt, std::array<double,2> bw) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    for (size_t i=0; i<2; ++i) {
        std::stringstream tmp;
        tmp << bw[i] << "GB/s";
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
    std::cout << std::setw(20) << "OpenMP bandwidth";
    std::cout << std::setw(20) << "OpenCL bandwidth";
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

float reduce_recursive_helper(OpenCL& opencl, cl::Buffer input, int n, int local_size, clock_type::time_point& copy_start_timer, clock_type::time_point& copy_end_timer) {
    cl::Buffer current_input = input;
    int current_n = n;

    cl::Kernel kernel(opencl.program, "reduce");

    while (true) {
        int num_groups = (current_n + local_size - 1) / local_size;
        int global_size = num_groups * local_size;
        cl::Buffer output(opencl.context, CL_MEM_READ_WRITE, num_groups * sizeof(float));

        kernel.setArg(0, current_input);
        kernel.setArg(1, output);
        kernel.setArg(2, cl::Local(local_size * sizeof(float)));
        kernel.setArg(3, current_n);

        opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));

        if (num_groups <= 1) {
            float result;
            opencl.queue.finish();
            copy_start_timer = clock_type::now();
            cl::copy(opencl.queue, output, &result, &result + 1);
            copy_end_timer = clock_type::now();
            return result;
        }
        current_input = output;
        current_n = num_groups;
    }
}

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

void profile_reduce(int n, OpenCL& opencl) {
    auto a = random_vector<float>(n);
    float result = 0, expected_result = 0;
    clock_type::time_point copy_start_timer;
    clock_type::time_point copy_end_timer;
    opencl.queue.flush();
    auto t0 = clock_type::now();
    expected_result = reduce(a);
    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
    opencl.queue.finish();
    auto t2 = clock_type::now();
    result = reduce_recursive_helper(opencl, d_a, n, 256, copy_start_timer, copy_end_timer);
    auto t3 = copy_start_timer;
    auto t4 = copy_end_timer;
    Vector<float> res_vec = {result};
    Vector<float> exp_vec = {expected_result};
    verify_vector(exp_vec, res_vec);

    print("reduce",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n+n, t0, t1), bandwidth(n*n+n+n, t2, t3)});
}

void profile_scan_inclusive(int n, OpenCL& opencl) {
    auto a = random_vector<float>(n);
    Vector<float> result(a), expected_result(a);
    auto t0 = clock_type::now();
    scan_inclusive(expected_result);
    auto t1 = clock_type::now();
    cl::Buffer d_buffer(opencl.queue, begin(a), end(a), true);
    opencl.queue.finish();
    auto t2 = clock_type::now();
    scan_recursive_helper(opencl, d_buffer, n, 256);
    opencl.queue.finish();
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_buffer, begin(result), end(result));
    auto t4 = clock_type::now();

    verify_vector(expected_result, result);
    print("scan-inclusive",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n, t0, t1), bandwidth(n, t2, t3)});
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_reduce(1024*1024*10, opencl);
    profile_scan_inclusive(1024*1024*10, opencl);
}

// const std::string src_before = R"(
// kernel void reduce(global float* a,
//                    global float* b,
//                    global float* result) {
//     // TODO: Implement OpenCL version.
// }

// kernel void scan_inclusive(global float* a,
//                            global float* b,
//                            global float* result) {
//     // TODO: Implement OpenCL version.
// }
// )";

const std::string src = R"(
kernel void reduce(global float* a,
                   global float* result,
                   local float* data,
                   int n) {
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

    // Редукция в локальной памяти
    for (int offset=local_size/2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            data[local_id] += data[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        result[group_id] = data[0];
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
